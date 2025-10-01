import asyncio
import re
import hashlib
import json
from typing import Optional, Dict
from rag_engine import RAGEngine
from context_manager import ContextManager
from llm_provider import llm_provider
from tools import book_ticket, cancel_ticket, get_my_tickets

# Shorter system prompt (<100 tokens)
SYSTEM_PROMPT = """You're Burraa's voice assistant for event booking. Be conversational, concise (1-2 sentences), and natural.

Guidelines:
- Use context and search results to answer accurately
- Never invent details
- For "that event" or "it", refer to recently mentioned events
- Keep it brief for voice delivery

You'll get context, search results, and the user's message."""

# Conversation States
STATE_AWAITING_PHONE = "AWAITING_PHONE"
STATE_CONVERSING = "CONVERSING"

# Response templates for simple intents (no LLM needed)
TEMPLATES = {
    "no_bookings": "You don't have any bookings yet. Want to explore some events?",
    "invalid_phone": "I need a 10-digit number. Can you try again?",
    "greeting": "Thank you! How may I assist you with events today?",
    "no_results": "No events found. Try 'concerts', 'food trails', or 'adventure'."
}

class ConversationManager:
    """Manages intelligent conversation with caching and optimization."""
    
    def __init__(self, transport, rag_engine: RAGEngine, redis_client):
        self.transport = transport
        self.rag = rag_engine
        self.state = STATE_AWAITING_PHONE
        self.phone_number = None
        self.redis_client = redis_client
        self.context_manager = None
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize(self):
        try:
            # Verify Redis (optional now)
            redis_available = False
            if self.redis_client:
                try:
                    redis_available = await self.redis_client.ping()
                except Exception as e:
                    print(f"⚠️ Redis unavailable: {e}. Running without caching.")
            
            # Initialize LLM provider
            await llm_provider.initialize()
            
            mode = "with caching" if redis_available else "stateless mode"
            print(f"✅ Connected to LLM ({mode})")
            await self.send_reply("Hello! I'm your event assistant. Could you share your 10-digit phone number?")
        
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            await self.send_reply("Something went wrong. Please try again!")
            if self.transport:
                await self.transport.close()
    
    async def cleanup(self):
        """Clean up resources."""
        await llm_provider.cleanup()
        if self.cache_hits + self.cache_misses > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            print(f"📊 Cache hit rate: {hit_rate:.1f}% ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
    
    async def send_reply(self, message: str):
        """Send complete message to client."""
        if not self.transport or self.transport.readyState != "open":
            return
        self.transport.send(f"[COMPLETE]{message}\n")
    
    async def send_chunk(self, chunk: str):
        """Send streaming chunk to client."""
        if not self.transport or self.transport.readyState != "open":
            return
        self.transport.send(f"[CHUNK]{chunk}")
    
    async def send_done(self):
        """Signal streaming completion."""
        if not self.transport or self.transport.readyState != "open":
            return
        self.transport.send("[DONE]\n")
    
    def _clean_message(self, message: str) -> str:
        """Clean up message."""
        return re.sub(r'\s+', ' ', message.strip())
    
    def extract_intent(self, message: str) -> Dict:
        """Extract user intent using patterns."""
        msg_lower = message.lower()
        
        # Cancel intent
        cancel_match = re.search(r'\b(cancel|delete|remove).*?(tic_\w+)', msg_lower)
        if cancel_match:
            return {"type": "cancel", "booking_id": cancel_match.group(2)}
        
        # Book intent with explicit event ID
        book_match = re.search(r'\b(book|buy|purchase|reserve|get).*?(evt\d+)', msg_lower)
        if book_match:
            quantity_match = re.search(r'\b(\d+)\s*(?:ticket|seat|spot)', msg_lower)
            quantity = int(quantity_match.group(1)) if quantity_match else 1
            return {"type": "book", "event_id": book_match.group(2), "quantity": quantity}
        
        # Book intent with reference
        if re.search(r'\b(book|buy|purchase|reserve|get)\s+(that|it|this)', msg_lower):
            resolved_id = self.context_manager.resolve_reference(message) if self.context_manager else None
            if resolved_id:
                quantity_match = re.search(r'\b(\d+)\s*(?:ticket|seat|spot)', msg_lower)
                quantity = int(quantity_match.group(1)) if quantity_match else 1
                return {"type": "book", "event_id": resolved_id, "quantity": quantity}
        
        # My tickets intent
        if re.search(r'\b(my|show|view|list|get).*?(ticket|booking)', msg_lower):
            return {"type": "my_tickets"}
        
        # Similar events intent
        if re.search(r'\b(similar|like|related)', msg_lower):
            resolved_id = self.context_manager.resolve_reference(message) if self.context_manager else None
            if resolved_id:
                return {"type": "similar", "event_id": resolved_id}
        
        # Details query
        if re.search(r'\b(price|cost|how much|details|info|tell me about)\b', msg_lower):
            if re.search(r'\b(that|it|this|first|second)', msg_lower):
                resolved_id = self.context_manager.resolve_reference(message) if self.context_manager else None
                if resolved_id:
                    return {"type": "details", "event_id": resolved_id}
        
        return {"type": "search", "query": message}
    
    def format_event_brief(self, event: Dict) -> str:
        """Format event in brief (token-optimized)."""
        return f"{event['name']} | {event.get('date/days', 'TBA')} | {event.get('location', 'TBA')} | {event.get('price', 'TBA')}"
    
    def format_search_results(self, results: list) -> str:
        """Format search results concisely."""
        if not results:
            return TEMPLATES["no_results"]
        
        formatted = "Found: "
        for i, event in enumerate(results[:2]):
            if i > 0:
                formatted += " Also, "
            formatted += f"{event['name']} on {event.get('date/days', 'TBA')} at {event.get('location', 'TBA')}. "
        return formatted
    
    async def handle_message(self, message: str):
        """Handle incoming messages with optimization."""
        message = self._clean_message(message)
        
        # Phone number collection
        if self.state == STATE_AWAITING_PHONE:
            if message.isdigit() and len(message) == 10:
                self.phone_number = message
                self.state = STATE_CONVERSING
                
                if self.redis_client:
                    self.context_manager = ContextManager(self.redis_client, self.phone_number)
                    await self.context_manager.load()
                
                await self.send_reply(TEMPLATES["greeting"])
            else:
                await self.send_reply(TEMPLATES["invalid_phone"])
            return
        
        # Add to context
        if self.context_manager:
            self.context_manager.add_message("user", message)
            await self.context_manager.save()
        
        # Extract intent
        intent = self.extract_intent(message)
        
        # Handle intents with templates (no LLM)
        if intent["type"] == "cancel":
            result = await cancel_ticket(intent["booking_id"], self.phone_number)
            await self.send_reply(f"{'Done! ' + result['message'] if result['status'] == 'success' else 'Hmm, ' + result['message']}")
            return
        
        elif intent["type"] == "book":
            result = await book_ticket(intent["event_id"], intent["quantity"], self.phone_number)
            if result["status"] == "success":
                booking = result["data"]
                response = (f"Booked! 🎉 {booking['event_name']} on {booking['event_date']}. "
                           f"Total: ₹{booking['total_price']} for {booking['quantity']} ticket(s). "
                           f"ID: {booking['booking_id']}")
                await self.send_reply(response)
                if self.context_manager:
                    self.context_manager.clear_pending_booking()
                    await self.context_manager.save()
            else:
                await self.send_reply(result["message"])
            return
        
        elif intent["type"] == "my_tickets":
            result = await get_my_tickets(self.phone_number)
            if result["status"] == "success":
                bookings = result["data"]
                response = "Your Bookings:\n" + "\n".join(
                    f"• {b['event_name']} - {b['event_date']} ({b['booking_id']})" for b in bookings
                )
                await self.send_reply(response)
            else:
                await self.send_reply(TEMPLATES["no_bookings"])
            return
        
        # Intents requiring LLM (search, similar, details)
        await self._handle_llm_intent(intent, message)
    
    async def _handle_llm_intent(self, intent: Dict, message: str):
        """Handle intents that need LLM response."""
        tool_result = None
        
        if intent["type"] == "similar":
            similar = await self.rag.find_similar_events(intent["event_id"], top_k=3)
            if similar:
                if self.context_manager:
                    self.context_manager.set_mentioned_events([e['id'] for e in similar])
                tool_result = self.format_search_results(similar)
            else:
                tool_result = "Couldn't find similar events."
        
        elif intent["type"] == "details":
            event = self.rag.get_event_by_id(intent["event_id"])
            if event:
                tool_result = f"{event['name']} on {event.get('date/days', 'TBA')} at {event.get('time', 'TBA')} in {event.get('location', 'TBA')}. Price: {event.get('price', 'TBA')}."
                if self.context_manager:
                    self.context_manager.set_mentioned_events([event['id']])
            else:
                tool_result = "Event not found."
        
        elif intent["type"] == "search":
            if self.context_manager:
                self.context_manager.set_last_search(message)
            results = await self.rag.search(message, top_k=5)
            if results:
                if self.context_manager:
                    self.context_manager.set_mentioned_events([e['id'] for e in results])
                tool_result = self.format_search_results(results)
            else:
                tool_result = TEMPLATES["no_results"]
        
        # Generate LLM response with caching
        await self._generate_cached_response(message, tool_result, intent)
        
        if self.context_manager:
            await self.context_manager.save()
    
    def _response_cache_key(self, message: str, tool_result: Optional[str], intent: Dict) -> str:
        """Generate cache key for LLM response."""
        context_str = ""
        if self.context_manager:
            context_str = str(self.context_manager.get_mentioned_events())
        
        hash_input = f"{intent['type']}:{message}:{tool_result}:{context_str}"
        return f"llm:response:{hashlib.md5(hash_input.encode()).hexdigest()}"
    
    async def _generate_cached_response(self, message: str, tool_result: Optional[str], intent: Dict):
        """Generate LLM response with Redis caching."""
        # Try cache
        if self.redis_client:
            try:
                cache_key = self._response_cache_key(message, tool_result, intent)
                cached = await self.redis_client.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    print(f"💾 LLM cache hit")
                    await self.send_reply(cached)
                    if self.context_manager:
                        self.context_manager.add_message("assistant", cached)
                    return
            except Exception as e:
                print(f"Cache read error: {e}")
        
        self.cache_misses += 1
        
        # Build messages (minimal context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Only add context if user references something ("that", "it")
        if self.context_manager and any(ref in message.lower() for ref in ['that', 'it', 'this', 'first', 'second']):
            context_summary = self.context_manager.build_context_summary()
            if context_summary != "No prior context":
                messages.append({"role": "system", "content": f"Context: {context_summary}"})
        
        # Add tool results
        if tool_result:
            messages.append({"role": "system", "content": f"Results: {tool_result}"})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Stream response
        full_response = ""
        try:
            async for chunk in llm_provider.generate(messages, max_tokens=120):
                full_response += chunk
                await self.send_chunk(chunk)
            
            await self.send_done()
            
            # Cache response
            if self.redis_client and full_response:
                try:
                    cache_key = self._response_cache_key(message, tool_result, intent)
                    await self.redis_client.set(cache_key, full_response, ex=300)  # 5min TTL
                except Exception as e:
                    print(f"Cache write error: {e}")
            
            # Save to context
            if self.context_manager:
                self.context_manager.add_message("assistant", full_response)
        
        except Exception as e:
            print(f"LLM error: {e}")
            fallback = tool_result if tool_result else "Sorry, I'm having trouble. Can you rephrase?"
            await self.send_reply(fallback)