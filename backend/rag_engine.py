import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class RAGEngine:
    """Fast semantic search using FAISS vector store."""
    
    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384-dim embeddings
        self.index = None
        self.events = []
        self.event_texts = []
        
    def build_index(self):
        """Build FAISS index at startup. Called once."""
        print("🔨 Building FAISS index...")
        
        # Load knowledge base
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.events = data.get('events', [])
        
        if not self.events:
            print("⚠️  No events found in knowledge base")
            return
        
        # Create searchable text for each event
        self.event_texts = []
        for event in self.events:
            text = f"{event.get('name', '')} {event.get('type', '')} {event.get('location', '')} {event.get('description', '')}"
            self.event_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(self.event_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built with {len(self.events)} events")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for events using semantic similarity."""
        if not self.index or not self.events:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.events)))
        
        # Return matching events with relevance scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.events):
                event = self.events[idx].copy()
                event['_relevance'] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append(event)
        
        # Filter by relevance threshold
        results = [r for r in results if r['_relevance'] > 0.3]
        return results
    
    def get_event_by_id(self, event_id: str) -> Dict:
        """Get specific event by ID."""
        for event in self.events:
            if event.get('id') == event_id:
                return event
        return None
    
    def find_similar_events(self, event_id: str, top_k: int = 3) -> List[Dict]:
        """Find events similar to a given event."""
        source_event = self.get_event_by_id(event_id)
        if not source_event:
            return []
        
        # Search using the source event's text
        source_idx = next((i for i, e in enumerate(self.events) if e.get('id') == event_id), -1)
        if source_idx == -1:
            return []
        
        query_text = self.event_texts[source_idx]
        results = self.search(query_text, top_k + 1)  # +1 to exclude self
        
        # Remove the source event itself
        return [r for r in results if r.get('id') != event_id][:top_k]