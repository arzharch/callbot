import redis.asyncio as redis
import json
import asyncio
import re
import aiohttp
from tools import knowledge_base_search, book_ticket, cancel_ticket, get_my_tickets, get_all_events

# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
OLLAMA_API_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'mistral'

# --- Conversation States ---
STATE_AWAITING_PHONE = "AWAITING_PHONE"
STATE_CONVERSING = "CONVERSING"

# System prompt for the AI
SYSTEM_PROMPT = """You are a helpful AI assistant for an event booking company. Your role is to:

1. Help users search for events using natural language
2. Book tickets when users want to attend events
3. Cancel bookings when requested
4. Show users their current bookings

Key guidelines:
- Be conversational, friendly, and concise (2-3 sentences max)
- When users ask about events, I will provide you with search results
- For booking, ask for confirmation if quantity is unclear
- Always acknowledge actions taken (bookings/cancellations)
- If the user's intent is unclear, ask a clarifying question
- Don't make up event details - only use information provided to you

You will receive tool results in your context. Respond naturally based on those results."""

class ConversationManager:
    """Manages conversation with LLM-powered natural language understanding."""
    
    def __init__(self, transport):
        self.transport = transport
        self.state = STATE_AWAITING_PHONE
        self.phone_number = None
        self.redis_client = None
        self.history = []
        self.session = None
        self.ollama_available = True  # Track Ollama availability

    async def initialize(self):
        try:
            # Initialize HTTP session first
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Connect to Redis
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            await self.redis_client.ping()
            
            print("Successfully connected to Redis and initialized HTTP session.")
            await self.send_reply("Hello! I'm your AI assistant for event bookings. Please provide your 10-digit phone number to begin.")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}")
            await self.send_reply("Error: Could not connect to the memory service. Please try again later.")
            if self.transport: 
                await self.transport.close()
        except Exception as e:
            print(f"Initialization error: {e}")
            await self.send_reply("Error: Failed to initialize the assistant. Please try again.")
            if self.transport:
                await self.transport.close()

    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            print("HTTP session closed.")

    async def call_ollama(self, user_message: str, tool_context: str = None) -> str:
        """Call Ollama Mistral API with conversation history."""
        
        # Ensure session exists
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add relevant history (last 6 messages to keep context manageable)
            for msg in self.history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add tool context if available
            if tool_context:
                messages.append({"role": "system", "content": f"Tool Result: {tool_context}"})
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150  # Keep responses concise
                }
            }
            
            async with self.session.post(OLLAMA_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['message']['content'].strip()
                else:
                    error_text = await response.text()
                    print(f"Ollama API error: {response.status} - {error_text}")
                    return "I'm having trouble thinking right now. Could you try rephrasing that?"
                    
        except asyncio.TimeoutError:
            print("Ollama timeout error")
            return "Sorry, I'm taking too long to respond. Please try again."
        except aiohttp.ClientConnectorError:
            print("Cannot connect to Ollama. Is it running?")
            self.ollama_available = False
            return "I can't connect to my brain right now. I'll use basic responses instead."
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "I encountered an error. Please try again."

    async def send_reply(self, message):
        """Sends a complete message to the client."""
        if not self.transport or self.transport.readyState != "open":
            print("Transport not open, cannot send message.")
            return
        self.transport.send(message + "\n")
        self.history.append({"role": "assistant", "content": message})
        await self.save_history()

    def extract_intent(self, message: str) -> dict:
        """Extract user intent using regex patterns for critical actions."""
        msg_lower = message.lower()
        
        # Cancel intent
        cancel_match = re.search(r'\b(cancel|delete|remove).*?(tic_\w+)', msg_lower)
        if cancel_match:
            return {"type": "cancel", "booking_id": cancel_match.group(2)}
        
        # Book intent with event ID
        book_match = re.search(r'\b(book|buy|purchase|reserve|get).*?(evt\d+)', msg_lower)
        if book_match:
            quantity_match = re.search(r'\b(\d+)\s*(?:ticket|seat)', msg_lower)
            quantity = int(quantity_match.group(1)) if quantity_match else 1
            return {"type": "book", "event_id": book_match.group(2), "quantity": quantity}
        
        # My tickets intent
        if re.search(r'\b(my|show|view|list|get).*?(ticket|booking)', msg_lower):
            return {"type": "my_tickets"}
        
        # Search intent (default)
        return {"type": "search", "query": message}

    def format_search_results(self, results: list) -> str:
        """Format search results for LLM context."""
        if not results:
            return "No events found matching the query."
        
        formatted = "Here are the matching events:\n"
        for event in results[:5]:  # Limit to top 5 results
            formatted += f"- {event['name']} (ID: {event['id']}) on {event['date']} - ${event['price']}\n"
        return formatted

    def format_tool_result(self, tool_result: dict) -> str:
        """Format tool results for LLM context."""
        if not isinstance(tool_result, dict):
            return "Unexpected error occurred."
        
        status = tool_result.get("status")
        
        if status == "success":
            data = tool_result.get("data")
            if isinstance(data, list):
                # Format list of events or tickets
                items = []
                for item in data:
                    parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in item.items()]
                    items.append(", ".join(parts))
                return "\n".join(items)
            elif isinstance(data, dict):
                # Format single booking
                parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in data.items()]
                return ", ".join(parts)
            return tool_result.get("message", "Action completed successfully.")
        
        elif status == "not_found":
            return tool_result.get("message", "No results found.")
        
        else:  # error
            return tool_result.get("message", "An error occurred.")

    def generate_simple_response(self, tool_context: str, intent_type: str) -> str:
        """Generate a simple response without LLM (fallback mode)."""
        if intent_type == "search":
            if "No matching events found" in tool_context or "No events found" in tool_context:
                return "I couldn't find any events matching your search. Try different keywords like 'concert', 'Mumbai', or 'workshop'."
            return f"I found some events for you:\n{tool_context}\n\nTo book, use the event ID like: 'book evt123'"
        
        elif intent_type == "book":
            if "success" in tool_context.lower():
                return f"Great! Your booking is confirmed. {tool_context}"
            else:
                return f"Sorry, couldn't complete the booking. {tool_context}"
        
        elif intent_type == "cancel":
            if "canceled" in tool_context.lower():
                return f"Done! {tool_context}"
            else:
                return f"Issue with cancellation: {tool_context}"
        
        elif intent_type == "my_tickets":
            if "no booked tickets" in tool_context.lower():
                return "You don't have any tickets booked yet. Search for events to get started!"
            return f"Here are your bookings:\n{tool_context}"
        
        return tool_context

    async def handle_message(self, message):
        """Handle user messages with LLM-powered understanding."""
        
        # Handle phone number collection
        if self.state == STATE_AWAITING_PHONE:
            if message.isdigit() and len(message) == 10:
                self.phone_number = message
                self.state = STATE_CONVERSING
                await self.load_history()
                await self.send_reply("Thank you! How can I help you today? You can ask me about events, book tickets, or check your bookings.")
            else:
                await self.send_reply("Please provide a valid 10-digit phone number.")
            return

        # --- Main Conversation Logic ---
        self.history.append({"role": "user", "content": message})
        await self.save_history()
        
        # Extract intent
        intent = self.extract_intent(message)
        tool_context = None
        
        # Execute appropriate tool based on intent
        if intent["type"] == "cancel":
            result = await cancel_ticket(intent["booking_id"], self.phone_number)
            tool_context = self.format_tool_result(result)
        
        elif intent["type"] == "book":
            result = await book_ticket(intent["event_id"], intent["quantity"], self.phone_number)
            tool_context = self.format_tool_result(result)
        
        elif intent["type"] == "my_tickets":
            result = await get_my_tickets(self.phone_number)
            tool_context = self.format_tool_result(result)
        
        elif intent["type"] == "search":
            # Perform smart search
            result = await knowledge_base_search(intent["query"])
            if result.get("status") == "success":
                tool_context = self.format_search_results(result.get("data", []))
            else:
                tool_context = "No matching events found. The user might want to try different search terms."
        
        # Get response - use LLM if available, otherwise use simple responses
        if self.ollama_available:
            response = await self.call_ollama(message, tool_context)
        else:
            response = self.generate_simple_response(tool_context, intent["type"])
        
        await self.send_reply(response)

    async def load_history(self):
        """Load conversation history from Redis."""
        if not self.redis_client or not self.phone_number:
            return
        try:
            data = await self.redis_client.get(self.phone_number)
            if data:
                self.history = json.loads(data)
        except Exception as e:
            print(f"Error loading history: {e}")

    async def save_history(self):
        """Save conversation history to Redis."""
        if not self.redis_client or not self.phone_number:
            return
        try:
            # Keep only last 20 messages to prevent memory bloat
            trimmed_history = self.history[-20:]
            await self.redis_client.set(
                self.phone_number, 
                json.dumps(trimmed_history), 
                ex=3600
            )
        except Exception as e:
            print(f"Error saving history: {e}")