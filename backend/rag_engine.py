import json
import numpy as np
import faiss
import hashlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os

class RAGEngine:
    """Fast semantic search using FAISS with Redis caching."""
    
    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.events = []
        self.event_texts = []
        self.redis_client = None
        
    def set_redis(self, redis_client):
        """Set Redis client for caching."""
        self.redis_client = redis_client
        
    def build_index(self):
        """Build FAISS index at startup."""
        print("🔨 Building FAISS index...")
        
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.events = data.get('events', [])
        
        if not self.events:
            print("⚠️  No events found in knowledge base")
            return
        
        # Create focused searchable text (name, type, location only)
        self.event_texts = []
        for event in self.events:
            text = f"{event.get('name', '')} {event.get('type', '')} {event.get('location', '')}"
            self.event_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(self.event_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built with {len(self.events)} events")
    
    def _cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for search query."""
        hash_input = f"{query.lower().strip()}:{top_k}"
        return f"rag:search:{hashlib.md5(hash_input.encode()).hexdigest()}"
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for events with Redis caching."""
        if not self.index or not self.events:
            return []
        
        # Try cache first
        if self.redis_client:
            try:
                cache_key = self._cache_key(query, top_k)
                cached = await self.redis_client.get(cache_key)
                if cached:
                    print(f"📦 Cache hit for query: {query[:30]}...")
                    return json.loads(cached)
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Perform search
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.events)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.events):
                event = self.events[idx].copy()
                event['_relevance'] = float(1 / (1 + dist))
                results.append(event)
        
        # Filter by relevance threshold
        results = [r for r in results if r['_relevance'] > 0.3]
        
        # Cache results
        if self.redis_client and results:
            try:
                cache_key = self._cache_key(query, top_k)
                await self.redis_client.set(cache_key, json.dumps(results), ex=900)  # 15min TTL
            except Exception as e:
                print(f"Cache write error: {e}")
        
        return results
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        """Get specific event by ID."""
        for event in self.events:
            if event.get('id') == event_id:
                return event
        return None
    
    async def find_similar_events(self, event_id: str, top_k: int = 3) -> List[Dict]:
        """Find events similar to a given event."""
        source_event = self.get_event_by_id(event_id)
        if not source_event:
            return []
        
        source_idx = next((i for i, e in enumerate(self.events) if e.get('id') == event_id), -1)
        if source_idx == -1:
            return []
        
        query_text = self.event_texts[source_idx]
        results = await self.search(query_text, top_k + 1)
        
        return [r for r in results if r.get('id') != event_id][:top_k]