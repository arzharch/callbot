import json
import aiofiles
import os
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- File Paths ---
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
TICKETS_PATH = os.path.join(os.path.dirname(__file__), 'tickets.json')

# --- Helper Functions ---
async def read_json_file(path):
    try:
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            return json.loads(content) if content else {}
    except FileNotFoundError:
        return {}

async def write_json_file(path, data):
    async with aiofiles.open(path, 'w') as f:
        await f.write(json.dumps(data, indent=2))

# --- Enhanced Search with Multiple Strategies ---

async def knowledge_base_search(query: str):
    """
    Performs intelligent search using multiple strategies:
    1. Exact ID match
    2. Keyword matching for type/location
    3. TF-IDF similarity for fuzzy matching
    """
    db = await read_json_file(KNOWLEDGE_BASE_PATH)
    events = db.get('events', [])
    
    if not events:
        return {"status": "not_found", "message": "The knowledge base is currently empty."}
    
    query_lower = query.lower()
    
    # Strategy 1: Check for exact event ID
    id_match = re.search(r'\b(evt\d+)\b', query_lower)
    if id_match:
        event_id = id_match.group(1)
        matching_event = next((e for e in events if e['id'].lower() == event_id), None)
        if matching_event:
            return {"status": "success", "data": [format_event_brief(matching_event)]}
    
    # Strategy 2: Keyword matching for common attributes
    keyword_matches = []
    for event in events:
        score = 0
        event_text = f"{event.get('name', '')} {event.get('type', '')} {event.get('location', '')} {event.get('date', '')}".lower()
        
        # Boost exact word matches
        query_words = set(query_lower.split())
        event_words = set(event_text.split())
        common_words = query_words & event_words
        score += len(common_words) * 2
        
        # Check for type matches
        event_type = event.get('type', '').lower()
        if event_type and event_type in query_lower:
            score += 5
        
        # Check for location matches
        location = event.get('location', '').lower()
        if location and location in query_lower:
            score += 3
        
        # Check for name matches (partial)
        name = event.get('name', '').lower()
        if name and any(word in name for word in query_lower.split()):
            score += 4
        
        if score > 0:
            keyword_matches.append((score, event))
    
    # If we have strong keyword matches, use those
    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[0], reverse=True)
        # Filter for reasonable threshold
        good_matches = [e for score, e in keyword_matches if score >= 3]
        if good_matches:
            return {"status": "success", "data": [format_event_brief(e) for e in good_matches[:5]]}
    
    # Strategy 3: TF-IDF similarity for fuzzy semantic matching
    documents = []
    for e in events:
        doc = f"{e.get('name', '')} {e.get('type', '')} {e.get('location', '')} {e.get('description', '')}"
        documents.append(doc)
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        threshold = 0.1  # Lower threshold for more permissive matching
        
        matching_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        
        if matching_indices:
            results = sorted(
                [(similarities[i], events[i]) for i in matching_indices],
                key=lambda x: x[0],
                reverse=True
            )
            matching_events = [event for score, event in results[:5]]
            return {"status": "success", "data": [format_event_brief(e) for e in matching_events]}
    
    except ValueError:
        pass  # Vocabulary might be empty
    
    # No matches found
    return {"status": "not_found", "message": "No events found. Try different search terms."}


def format_event_brief(event: dict) -> dict:
    """Format event data for brief display."""
    return {
        "id": event["id"],
        "name": event["name"],
        "date": event["date"],
        "price": f"${event['price']}",
        "location": event.get("location", "TBA"),
        "tickets_available": event.get("tickets_available", 0)
    }


async def get_all_events():
    """Get all events from knowledge base."""
    db = await read_json_file(KNOWLEDGE_BASE_PATH)
    events = db.get('events', [])
    return {"status": "success", "data": [format_event_brief(e) for e in events]}


# --- Booking Tools ---

async def book_ticket(event_id: str, quantity: int, phone_number: str):
    """Book tickets for an event."""
    kb_db = await read_json_file(KNOWLEDGE_BASE_PATH)
    events = {event['id']: event for event in kb_db.get('events', [])}
    
    if event_id not in events:
        return {"status": "error", "message": f"Event ID '{event_id}' not found."}
    
    event = events[event_id]
    
    if event['tickets_available'] < quantity:
        return {
            "status": "error",
            "message": f"Only {event['tickets_available']} tickets left for {event['name']}."
        }
    
    # Update available tickets
    for e in kb_db['events']:
        if e['id'] == event_id:
            e['tickets_available'] -= quantity
            break
    
    await write_json_file(KNOWLEDGE_BASE_PATH, kb_db)
    
    # Create booking
    tickets_db = await read_json_file(TICKETS_PATH)
    booking_id = f"tic_{int(datetime.now().timestamp())}"
    booking = {
        "booking_id": booking_id,
        "event_id": event_id,
        "event_name": event['name'],
        "quantity": quantity,
        "total_price": event['price'] * quantity,
        "booked_at": datetime.now().isoformat()
    }
    
    if phone_number not in tickets_db:
        tickets_db[phone_number] = []
    tickets_db[phone_number].append(booking)
    
    await write_json_file(TICKETS_PATH, tickets_db)
    
    return {"status": "success", "data": booking}


async def cancel_ticket(booking_id: str, phone_number: str):
    """Cancel a ticket booking."""
    tickets_db = await read_json_file(TICKETS_PATH)
    user_tickets = tickets_db.get(phone_number, [])
    
    ticket_to_cancel = next((t for t in user_tickets if t.get('booking_id') == booking_id), None)
    
    if not ticket_to_cancel:
        return {"status": "error", "message": f"Booking ID '{booking_id}' not found."}
    
    # Remove from user's tickets
    user_tickets.remove(ticket_to_cancel)
    tickets_db[phone_number] = user_tickets
    
    if not user_tickets:
        del tickets_db[phone_number]
    
    await write_json_file(TICKETS_PATH, tickets_db)
    
    # Restore tickets to event
    kb_db = await read_json_file(KNOWLEDGE_BASE_PATH)
    for event in kb_db.get('events', []):
        if event['id'] == ticket_to_cancel['event_id']:
            event['tickets_available'] += ticket_to_cancel['quantity']
            break
    
    await write_json_file(KNOWLEDGE_BASE_PATH, kb_db)
    
    return {
        "status": "success",
        "message": f"Booking for '{ticket_to_cancel['event_name']}' (ID: {booking_id}) is canceled."
    }


async def get_my_tickets(phone_number: str):
    """Get all tickets for a user."""
    tickets_db = await read_json_file(TICKETS_PATH)
    user_tickets = tickets_db.get(phone_number, [])
    
    if not user_tickets:
        return {"status": "not_found", "message": "You have no booked tickets."}
    
    return {"status": "success", "data": user_tickets}