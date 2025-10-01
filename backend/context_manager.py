import json
from typing import List, Dict, Optional
from datetime import datetime

class ContextManager:
    """Manages session context for natural conversations with minimal overhead."""
    
    def __init__(self, redis_client, phone_number: str):
        self.redis = redis_client
        self.phone = phone_number
        self.context = {
            "last_mentioned_events": [],  # IDs of recently discussed events (max 3)
            "pending_booking": None,
            "last_search_query": None,
            "conversation_history": []  # Last 6 turns only
        }
    
    async def load(self):
        """Load context from Redis."""
        if not self.redis:
            return
        
        try:
            key = f"context:{self.phone}"
            data = await self.redis.get(key)
            if data:
                self.context = json.loads(data)
        except Exception as e:
            print(f"Context load error: {e}")
    
    async def save(self):
        """Save context to Redis with trimming."""
        if not self.redis:
            return
        
        try:
            # Trim history to last 6 turns
            if len(self.context["conversation_history"]) > 6:
                self.context["conversation_history"] = self.context["conversation_history"][-6:]
            
            # Trim mentioned events to last 3
            if len(self.context["last_mentioned_events"]) > 3:
                self.context["last_mentioned_events"] = self.context["last_mentioned_events"][:3]
            
            key = f"context:{self.phone}"
            await self.redis.set(key, json.dumps(self.context), ex=3600)  # 1 hour TTL
        except Exception as e:
            print(f"Context save error: {e}")
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.context["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def set_mentioned_events(self, event_ids: List[str]):
        """Update recently mentioned events (keep top 3)."""
        self.context["last_mentioned_events"] = event_ids[:3]
    
    def get_mentioned_events(self) -> List[str]:
        """Get recently mentioned event IDs."""
        return self.context.get("last_mentioned_events", [])
    
    def set_pending_booking(self, event_id: str, quantity: int = 1):
        """Set a pending booking action."""
        self.context["pending_booking"] = {
            "event_id": event_id,
            "quantity": quantity
        }
    
    def get_pending_booking(self) -> Optional[Dict]:
        """Get pending booking if any."""
        return self.context.get("pending_booking")
    
    def clear_pending_booking(self):
        """Clear pending booking after completion."""
        self.context["pending_booking"] = None
    
    def set_last_search(self, query: str):
        """Record last search query."""
        self.context["last_search_query"] = query
    
    def get_last_search(self) -> Optional[str]:
        """Get last search query."""
        return self.context.get("last_search_query")
    
    def get_conversation_history(self, limit: int = 6) -> List[Dict]:
        """Get recent conversation history."""
        history = self.context.get("conversation_history", [])
        return history[-limit:]
    
    def resolve_reference(self, text: str) -> Optional[str]:
        """
        Resolve references like 'that event', 'the first one', 'it'.
        Returns event_id if resolvable.
        """
        text_lower = text.lower()
        mentioned = self.get_mentioned_events()
        
        if not mentioned:
            return None
        
        # Common reference patterns
        if any(ref in text_lower for ref in ['that', 'this', 'it', 'that one', 'this one']):
            return mentioned[0]  # Most recent
        
        if 'first' in text_lower and len(mentioned) >= 1:
            return mentioned[0]
        
        if 'second' in text_lower and len(mentioned) >= 2:
            return mentioned[1]
        
        if 'last' in text_lower and len(mentioned) >= 1:
            return mentioned[-1]
        
        return None
    
    def build_context_summary(self) -> str:
        """Build a minimal context summary for LLM (token-optimized)."""
        parts = []
        
        # Recent events (compact format)
        if self.context.get("last_mentioned_events"):
            event_list = ", ".join(self.context["last_mentioned_events"][:2])  # Only top 2
            parts.append(f"Recent events: {event_list}")
        
        # Pending booking
        if self.context.get("pending_booking"):
            pending = self.context["pending_booking"]
            parts.append(f"Considering: {pending['event_id']} (qty:{pending['quantity']})")
        
        return " | ".join(parts) if parts else "No prior context"