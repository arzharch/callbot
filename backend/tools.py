import json
import aiofiles
import os
from datetime import datetime
import re

# --- File Paths ---
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), 'burraa_catalog.txt')
BOOKINGS_PATH = os.path.join(os.path.dirname(__file__), 'bookings.json')

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

def parse_catalog_to_json():
    """Parse the catalog text file into structured JSON."""
    events = []
    
    try:
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by separator
        records = content.split('--------------------------------------------------------------------------------')
        
        event_id_counter = 1
        for record in records:
            if not record.strip() or 'BURRAA_CATALOG' in record:
                continue
            
            lines = [line.strip() for line in record.strip().split('\n') if line.strip()]
            
            event = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    event[key] = value
            
            if event.get('name'):
                event['id'] = f"evt{event_id_counter:03d}"
                event['tickets_available'] = 50  # Default availability
                events.append(event)
                event_id_counter += 1
        
        return {"events": events}
    
    except Exception as e:
        print(f"Error parsing catalog: {e}")
        return {"events": []}

# --- Booking Tools ---

async def book_ticket(event_id: str, quantity: int, phone_number: str):
    """Book tickets for an event."""
    # Load events from parsed catalog
    kb_data = parse_catalog_to_json()
    events = {event['id']: event for event in kb_data.get('events', [])}
    
    if event_id not in events:
        return {"status": "error", "message": f"Event ID '{event_id}' not found."}
    
    event = events[event_id]
    
    if event.get('tickets_available', 0) < quantity:
        return {
            "status": "error",
            "message": f"Only {event.get('tickets_available', 0)} tickets left for {event['name']}."
        }
    
    # Create booking
    bookings_db = await read_json_file(BOOKINGS_PATH)
    booking_id = f"tic_{int(datetime.now().timestamp())}"
    
    # Extract price (remove currency symbols)
    price_str = event.get('price', '₹0')
    price_match = re.search(r'[\d,]+', price_str.replace('₹', '').replace(',', ''))
    price = int(price_match.group()) if price_match else 0
    
    booking = {
        "booking_id": booking_id,
        "event_id": event_id,
        "event_name": event['name'],
        "event_date": event.get('date/days', 'TBA'),
        "event_time": event.get('time', 'TBA'),
        "location": event.get('location', 'TBA'),
        "quantity": quantity,
        "price_per_ticket": price,
        "total_price": price * quantity,
        "booked_at": datetime.now().isoformat(),
        "status": "confirmed"
    }
    
    if phone_number not in bookings_db:
        bookings_db[phone_number] = []
    bookings_db[phone_number].append(booking)
    
    await write_json_file(BOOKINGS_PATH, bookings_db)
    
    return {"status": "success", "data": booking}


async def cancel_ticket(booking_id: str, phone_number: str):
    """Cancel a ticket booking."""
    bookings_db = await read_json_file(BOOKINGS_PATH)
    user_bookings = bookings_db.get(phone_number, [])
    
    ticket_to_cancel = next((b for b in user_bookings if b.get('booking_id') == booking_id), None)
    
    if not ticket_to_cancel:
        return {"status": "error", "message": f"Booking ID '{booking_id}' not found."}
    
    # Remove from user's bookings
    user_bookings.remove(ticket_to_cancel)
    bookings_db[phone_number] = user_bookings
    
    if not user_bookings:
        del bookings_db[phone_number]
    
    await write_json_file(BOOKINGS_PATH, bookings_db)
    
    return {
        "status": "success",
        "message": f"Booking for '{ticket_to_cancel['event_name']}' (ID: {booking_id}) has been canceled."
    }


async def get_my_tickets(phone_number: str):
    """Get all tickets for a user."""
    bookings_db = await read_json_file(BOOKINGS_PATH)
    user_bookings = bookings_db.get(phone_number, [])
    
    if not user_bookings:
        return {"status": "not_found", "message": "You have no bookings yet."}
    
    return {"status": "success", "data": user_bookings}


def format_event_brief(event: dict) -> dict:
    """Format event data for brief display."""
    return {
        "id": event["id"],
        "name": event["name"],
        "type": event.get("type", "Event"),
        "date": event.get("date/days", "TBA"),
        "time": event.get("time", "TBA"),
        "price": event.get("price", "TBA"),
        "location": event.get("location", "TBA")
    }