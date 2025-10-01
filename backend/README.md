# 🎭 Burraa Event Chatbot

A lightning-fast, context-aware conversational AI chatbot for event discovery and booking. Built with FAISS RAG, Redis context management, and dual LLM support (Ollama/Gemini).

## ✨ Features

- **🚀 Fast Semantic Search**: FAISS-powered vector search (<10ms queries)
- **🧠 Context-Aware**: Remembers conversation history and resolves references
- **💬 Human-like Responses**: Natural, conversational tone with typing indicators
- **🔄 Dual LLM Support**: Switch between Ollama (local) and Gemini (cloud)
- **📱 Real-time Communication**: WebRTC-based messaging
- **🎯 Smart Intent Detection**: Handles bookings, cancellations, searches, and follow-ups

## 🏗️ Architecture

```
User → WebRTC → Bot Logic → [Intent Detection]
                              ↓
                    ┌─────────┴──────────┐
                    ↓                    ↓
              [Rule-based]        [Context + RAG]
                    ↓                    ↓
              Quick Reply        FAISS Search → LLM → Response
                    ↓                    ↓
                    └─────────┬──────────┘
                              ↓
                        Redis (Context)
```

### Key Components

1. **RAG Engine** (`rag_engine.py`)
   - FAISS in-memory vector store
   - SentenceTransformer embeddings (all-MiniLM-L6-v2)
   - Semantic similarity search

2. **Context Manager** (`context_manager.py`)
   - Session-based context tracking
   - Reference resolution ("that event", "the first one")
   - Conversation history (10 turns)

3. **LLM Provider** (`llm_provider.py`)
   - Unified interface for Ollama & Gemini
   - Configurable via environment variables
   - Fallback handling

4. **Bot Logic** (`bot_logic.py`)
   - Intent extraction
   - Typing indicators for slow operations
   - Natural conversation flow

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis
- Ollama (or Gemini API key)

### Installation

```bash
# Clone repository
cd backend

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### Running the Server

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Ollama (if using local LLM)
ollama serve
ollama pull mistral

# Terminal 3: Start chatbot server
source venv/bin/activate
python main.py
```

Server runs at: `http://localhost:8080`

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM Provider
LLM_PROVIDER=ollama  # or "gemini"

# Ollama (local)
OLLAMA_API_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=mistral

# Gemini (cloud)
GEMINI_API_KEY=your_api_key_here

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Switching to Gemini

1. Get API key: https://makersuite.google.com/app/apikey
2. Edit `.env`:
   ```bash
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_actual_key
   ```
3. Restart server

## 💡 Usage Examples

### User Conversations

**Example 1: Simple Search**
```
User: Show me concerts in Goa
Bot: I found 2 concert-related events in Goa: Rockfest Mayhem II and Saturday Night Fever. Want details?
```

**Example 2: Context-Aware**
```
User: Find food events
Bot: Here are 3 food trails: Feni & Tapas, Secret Food Tavern, and Margao City Food Trail.

User: What's the price of the first one?
Bot: Feni & Tapas Food Trail costs ₹3199 onwards. Age 21+ required. Want to book?
```

**Example 3: Smart Booking**
```
User: Book that for 2 people
Bot: 💭 Booking that for you...
Bot: Booked! 🎉 Feni & Tapas Food Trail on Evenings. Total: ₹6398 for 2 tickets. Booking ID: tic_1234567890
```

**Example 4: Similar Events**
```
User: Any similar events?
Bot: 💭 Finding similar events...
Bot: Found 3 similar options: Margao City Food Trail, Secret Food Tavern Trail, and Pizza Making Workshop. Want to know more?
```

## 🧪 Testing

### Test Search
```bash
curl -X POST http://localhost:8080/test \
  -H "Content-Type: application/json" \
  -d '{"query": "concerts in goa"}'
```

### Test Booking Flow
1. Connect via WebRTC client
2. Send: `9876543210` (phone number)
3. Send: `show me concerts`
4. Send: `book evt001 for 2 tickets`
5. Send: `my bookings`

## 📁 Project Structure

```
backend/
├── main.py                  # WebRTC server + startup
├── bot_logic.py             # Conversation manager
├── rag_engine.py            # FAISS vector search
├── context_manager.py       # Session context
├── llm_provider.py          # LLM abstraction
├── tools.py                 # Booking operations
├── burraa catalog.txt       # Event catalog (source)
├── knowledge_base.json      # Generated JSON (auto-created)
├── bookings.json            # User bookings
├── requirements.txt         # Dependencies
├── .env.example             # Config template
└── setup.sh                 # Setup script
```

## 🔧 Customization

### Adjust Response Speed

Edit `bot_logic.py`:
```python
# Faster responses (less context)
history = self.context_manager.get_conversation_history(limit=2)

# Slower but smarter (more context)
history = self.context_manager.get_conversation_history(limit=8)
```

### Change Typing Indicators

```python
# In bot_logic.py
await self.send_typing_indicator("Hmm, let me look into that...")
await self.send_typing_indicator("Give me a sec...")
```

### Modify Search Relevance

```python
# In rag_engine.py - stricter matching
results = [r for r in results if r['_relevance'] > 0.5]

# In rag_engine.py - more permissive
results = [r for r in results if r['_relevance'] > 0.2]
```

## 🐛 Troubleshooting

### FAISS Build Fails
```bash
# Use CPU-only version
pip uninstall faiss-gpu
pip install faiss-cpu==1.7.4
```

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### Redis Connection Error

```bash
# Check Redis
redis-cli ping

# Should return: PONG

# Restart Redis
redis-server
```

**Note on Docker:** If you are running the application inside a Docker container and Redis is running on your host machine, set the `REDIS_HOST` environment variable to `host.docker.internal`.

### Slow LLM Responses
1. Switch to Gemini (faster for complex queries)
2. Reduce context window in `bot_logic.py`
3. Use smaller Ollama model: `ollama pull mistral:7b-instruct-q4_K_M`

## 📊 Performance Benchmarks

- FAISS search: <10ms
- Context retrieval: <5ms
- Ollama (Mistral): ~1-2s
- Gemini (Flash): ~0.5-1s
- Rule-based replies: <50ms

## 🔐 Security Notes

- Phone numbers stored temporarily (1 hour TTL)
- No authentication in MVP (add JWT for production)
- WebRTC connections are P2P
- Redis data expires after 1 hour

## 🚀 Production Deployment

### Recommended Stack
- **Server**: Ubuntu 22.04 LTS
- **LLM**: Gemini 2.5 Flash (faster, scalable)
- **Redis**: Managed service (AWS ElastiCache, Redis Cloud)
- **Vector DB**: Pinecone or Weaviate (if scaling beyond FAISS)