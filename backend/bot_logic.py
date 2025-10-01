import redis.asyncio as redis
import asyncio
import re
from typing import Optional, Dict
from rag_engine import RAGEngine
from context_manager import ContextManager
from llm_provider import llm_provider
from tools import book_ticket, cancel_ticket, get_my_tickets, format_event_brief

# System prompt for the AI
SYSTEM_PROMPT = """You are Burraa's friendly event assistant. Keep responses natural and conversational (2-3 sentences max).

Guidelines:
- Be casual and human-like ("Sure thing!", "Let me check that out", "Hmm, interesting")
- Use the context and search results provided to answer accurately
- If asked about price/details of "that event" or "it", check the context for recently mentioned events
- For booking confirmations, be enthusiastic but brief
- If unsure, ask for clarification naturally
- Never make up event details - only use provided information
- Keep it short and sweet - users prefer quick replies

You'll receive:
1. Recent conversation context
2. Search results from our event database
3. User's current message

Respond naturally as if texting a friend."""

# Conversation States
STATE_AWAITING_PHONE = "AWAITING_PHONE"
STATE_CONVERSING = "CONVERSING"

class ConversationManager:
    """Manages intelligent conversation with context awareness."""
    
    def __init__(self, transport, rag_engine: RAGEngine, redis_client):
        self.transport = transport
        self.rag = rag_engine
        self.state = STATE_AWAITING_PHONE
        self.phone_number = None
        self.redis_client = redis_client
        self.context_manager = None
        
    async def initialize(self):
        try:
            # Verify Redis connection
            if not self.redis_client or not await self.redis_client.ping():
                raise redis.exceptions.ConnectionError("Redis client not available")

            # Initialize LLM provider
            await llm_provider.initialize()
            
            print("✅ Connected to Redis and LLM")
            await self.send_reply("Hey! I'm here to help you find and book amazing events. What's your phone number?")
        
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis connection error: {e}")
            await self.send_reply("Oops, having connection issues. Try again in a bit?")
            if self.transport:
                await self.transport.close()
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            await self.send_reply("Something went wrong on my end. Please try again!")
            if self.transport:
                await self.transport.close()
    
    async def cleanup(self):
        """Clean up resources."""
        await llm_provider.cleanup()
        print("🧹 Cleaned up resources")
    
    async def send_reply(self, message: str):
        """Send message to client."""
        if not self.transport or self.transport.readyState != "open":
            print("⚠️  Transport not open")
            return
        
        self.transport.send(message + "\n")
        
        # Save to context if available
        if self.context_manager:
            self.context_manager.add_message("assistant", message)
            await self.context_manager.save()
    
    async def send_typing_indicator(self, message: str = "Let me check that..."):
        """Send typing indicator for slow operations."""
        await self.send_reply(f"💭 {message}")
        await asyncio.sleep(0.3)  # Brief pause for natural feel
    
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
        
        # Book intent with reference ("book that", "book it")
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
        if re.search(r'\b(similar|like|related|comparable)', msg_lower):
            resolved_id = self.context_manager.resolve_reference(message) if self.context_manager else None
            if resolved_id:
                return {"type": "similar", "event_id": resolved_id}
        
        # Price/details query about referenced event
        if re.search(r'\b(price|cost|how much|details|info|tell me about)\b', msg_lower):
            if re.search(r'\b(that|it|this|first|second)', msg_lower):
                resolved_id = self.context_manager.resolve_reference(message) if self.context_manager else None
                if resolved_id:
                    return {"type": "details", "event_id": resolved_id}
        
        # Default: search
        return {"type": "search", "query": message}
    
    def format_search_results(self, results: list) -> str:
        """Format search results for LLM."""
        if not results:
            return "No events found."
        
        formatted = "Search Results:\n"
        for i, event in enumerate(results[:5], 1):
            formatted += f"{i}. {event['name']} ({event['id']}) - {event.get('date/days', 'TBA')}, {event.get('location', 'TBA')}, {event.get('price', 'TBA')}\n"
        return formatted
    
    def format_event_details(self, event: Dict) -> str:
        """Format single event details."""
        if not event:
            return "Event not found."
        
        return (f"Event: {event['name']} ({event['id']})\n"
                f"Type: {event.get('type', 'N/A')}\n"
                f"Date: {event.get('date/days', 'TBA')}\n"
                f"Time: {event.get('time', 'TBA')}\n"
                f"Location: {event.get('location', 'TBA')}\n"
                f"Price: {event.get('price', 'TBA')}\n"
                f"Description: {event.get('description', 'N/A')}")
    
    async def handle_message(self, message: str):
        """Handle incoming messages with intelligence."""
        
        # Phone number collection
        if self.state == STATE_AWAITING_PHONE:
            if message.isdigit() and len(message) == 10:
                self.phone_number = message
                self.state = STATE_CONVERSING
                
                # Initialize context manager
                self.context_manager = ContextManager(self.redis_client, self.phone_number)
                await self.context_manager.load()
                
                await self.send_reply("Perfect! What kind of events are you into? Concerts, food trails, adventure?")
            else:
                await self.send_reply("I need a 10-digit number. Can you try again?")
            return
        
        # Main conversation logic
        self.context_manager.add_message("user", message)
        await self.context_manager.save()
        
        # Extract intent
        intent = self.extract_intent(message)
        tool_result = None
        needs_llm = True
        
        # Handle different intents
        if intent["type"] == "cancel":
            result = await cancel_ticket(intent["booking_id"], self.phone_number)
            if result["status"] == "success":
                await self.send_reply(f"Done! {result['message']}")
                needs_llm = False
            else:
                await self.send_reply(f"Hmm, {result['message']}")
                needs_llm = False
        
        elif intent["type"] == "book":
            await self.send_typing_indicator("Booking that for you...")
            result = await book_ticket(intent["event_id"], intent["quantity"], self.phone_number)
            
            if result["status"] == "success":
                booking = result["data"]
                response = (f"Booked! 🎉 {booking['event_name']} on {booking['event_date']}. "
                           f"Total: ₹{booking['total_price']} for {booking['quantity']} ticket(s). "
                           f"Booking ID: {booking['booking_id']}")
                await self.send_reply(response)
                self.context_manager.clear_pending_booking()
                await self.context_manager.save()
                needs_llm = False
            else:
                tool_result = result["message"]
        
        elif intent["type"] == "my_tickets":
            result = await get_my_tickets(self.phone_number)
            if result["status"] == "success":
                bookings = result["data"]
                response = "Your Bookings:\n"
                for b in bookings:
                    response += f"• {b['event_name']} - {b['event_date']} ({b['booking_id']})\n"
                await self.send_reply(response)
                needs_llm = False
            else:
                await self.send_reply("You don't have any bookings yet. Want to explore some events?")
                needs_llm = False
        
        elif intent["type"] == "similar":
            await self.send_typing_indicator("Finding similar events...")
            similar = self.rag.find_similar_events(intent["event_id"], top_k=3)
            if similar:
                self.context_manager.set_mentioned_events([e['id'] for e in similar])
                tool_result = self.format_search_results(similar)
            else:
                tool_result = "Couldn't find similar events."
        
        elif intent["type"] == "details":
            event = self.rag.get_event_by_id(intent["event_id"])
            if event:
                tool_result = self.format_event_details(event)
                self.context_manager.set_mentioned_events([event['id']])
            else:
                tool_result = "Event not found."
        
        elif intent["type"] == "search":
            # Determine if search is complex (needs typing indicator)
            is_complex = len(message.split()) > 5 or "similar" in message.lower()
            
            if is_complex:
                await self.send_typing_indicator("Searching for you...")
            
            self.context_manager.set_last_search(message)
            results = self.rag.search(message, top_k=5)
            
            if results:
                self.context_manager.set_mentioned_events([e['id'] for e in results])
                tool_result = self.format_search_results(results)
            else:
                tool_result = "No events found. Try 'concerts', 'food trails', or 'adventure'."
        
        # Generate LLM response if needed
        if needs_llm:
            await self._generate_llm_response(message, tool_result)
        
        await self.context_manager.save()
    
    async def _generate_llm_response(self, user_message: str, tool_context: Optional[str]):
        """Generate response using LLM."""
        try:
            # Build messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add context summary
            context_summary = self.context_manager.build_context_summary()
            if context_summary and context_summary != "No prior context":
                messages.append({"role": "system", "content": f"Context: {context_summary}"})
            
            # Add recent history (last 4 turns)
            history = self.context_manager.get_conversation_history(limit=4)
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add tool results
            if tool_context:
                messages.append({"role": "system", "content": f"Tool Results:\n{tool_context}"})
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = await llm_provider.generate(messages, max_tokens=150)
            await self.send_reply(response)
        
        except Exception as e:
            print(f"LLM error: {e}")
            # Fallback to simple response
            if tool_context:
                await self.send_reply(tool_context)
            else:
                await self.send_reply("Sorry, I'm having trouble understanding. Can you rephrase that?")