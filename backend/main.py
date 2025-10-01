import asyncio
import json
import os
import uuid
import redis.asyncio as redis
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from bot_logic import ConversationManager
from rag_engine import RAGEngine
from tools import parse_catalog_to_json

# --- Setup ---
ROOT = os.path.dirname(__file__)
FRONTEND_PATH = os.path.abspath(os.path.join(ROOT, '../frontend'))
KNOWLEDGE_BASE_PATH = os.path.join(ROOT, 'burraa_catalog.txt')
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# Global RAG engine and Redis client (initialized once at startup)
rag_engine = None
redis_client = None
pcs = set()
conversation_managers = {}

# --- Startup Initialization ---

async def init_rag():
    """Initialize RAG engine at startup."""
    global rag_engine
    
    print("🚀 Initializing RAG engine...")
    
    # Parse catalog and save as JSON for RAG
    kb_data = parse_catalog_to_json()
    json_path = os.path.join(ROOT, 'knowledge_base.json')
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, indent=2, ensure_ascii=False)
    
    # Build FAISS index
    rag_engine = RAGEngine(json_path)
    rag_engine.build_index()
    
    print("✅ RAG engine ready!")

async def init_redis(app):
    """Initialize Redis client at startup."""
    global redis_client
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        await redis_client.ping()
        app['redis_client'] = redis_client
        print("✅ Connected to Redis!")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Redis connection error: {e}")
        print("⚠️  Context management will be disabled.")

# --- HTTP Route Handlers ---

async def index(request):
    """Serves the main index.html file."""
    try:
        with open(os.path.join(FRONTEND_PATH, "index.html"), "r") as f:
            return web.Response(content_type="text/html", text=f.read())
    except FileNotFoundError:
        return web.Response(status=404, text="index.html not found")

async def javascript(request):
    """Serves the client.js file."""
    try:
        with open(os.path.join(FRONTEND_PATH, "client.js"), "r") as f:
            return web.Response(content_type="application/javascript", text=f.read())
    except FileNotFoundError:
        return web.Response(status=404, text="client.js not found")

async def stylesheet(request):
    """Serves the style.css file."""
    try:
        with open(os.path.join(FRONTEND_PATH, "style.css"), "r") as f:
            return web.Response(content_type="text/css", text=f.read())
    except FileNotFoundError:
        return web.Response(status=404, text="style.css not found")

async def offer(request):
    """Handles the WebRTC offer from the client."""
    try:
        params = await request.json()
        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    except (json.JSONDecodeError, KeyError):
        return web.Response(status=400, text="Invalid offer format")

    pc = RTCPeerConnection()
    pc_id = f"PeerConnection({uuid.uuid4()})"
    pcs.add(pc)

    def log_info(msg, *args):
        print(f"{pc_id} {msg % args}")

    log_info("Created for %s", request.remote)

    # Create a conversation manager with shared RAG engine and Redis client
    conversation_manager = ConversationManager(
        transport=None, 
        rag_engine=rag_engine, 
        redis_client=request.app.get('redis_client')
    )
    conversation_managers[pc_id] = conversation_manager

    @pc.on("datachannel")
    def on_datachannel(channel):
        log_info(f"DataChannel '{channel.label}' created")
        conversation_manager.transport = channel

        @channel.on("open")
        async def on_open():
            log_info(f"DataChannel '{channel.label}' is open")
            await conversation_manager.initialize()

        @channel.on("message")
        async def on_message(message):
            log_info(f"Message from client: {message}")
            await conversation_manager.handle_message(message)
        
        @channel.on("close")
        def on_close():
            log_info(f"DataChannel '{channel.label}' closed")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            # Cleanup conversation manager
            if pc_id in conversation_managers:
                await conversation_managers[pc_id].cleanup()
                del conversation_managers[pc_id]
            
            await pc.close()
            pcs.discard(pc)
            log_info("PeerConnection closed and removed")

    try:
        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as e:
        log_info(f"Error during WebRTC negotiation: {e}")
        if pc_id in conversation_managers:
            await conversation_managers[pc_id].cleanup()
            del conversation_managers[pc_id]
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=500, text="WebRTC negotiation failed")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )

async def on_shutdown(app):
    """Closes all active peer connections and cleans up resources on shutdown."""
    # Cleanup all conversation managers
    cleanup_tasks = [manager.cleanup() for manager in conversation_managers.values()]
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    conversation_managers.clear()
    
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    # Close Redis client
    if redis_client:
        await redis_client.close()
        print("🔌 Redis client closed.")

    print("🧹 All peer connections closed and resources cleaned up.")

async def on_startup(app):
    """Initialize services on server startup."""
    await init_rag()
    await init_redis(app)

# --- Main Application Setup ---

if __name__ == "__main__":
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # Setup routes
    app.router.add_post("/offer", offer, name='offer')
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/style.css", stylesheet)

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "http://localhost:5173": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST"],
        ),
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=False,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST"],
        )
    })

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    print("=" * 60)
    print("🎭 BURRAA EVENT CHATBOT SERVER")
    print("=" * 60)
    print("📍 Server: http://0.0.0.0:8080")
    print("🤖 LLM Provider: Check environment (OLLAMA/GEMINI)")
    print("=" * 60)
    print("\n⚙️  Pre-flight Checklist:")
    print("   ✓ Redis running? (redis-server)")
    print("   ✓ Ollama running? (ollama serve)")
    print("   ✓ Mistral pulled? (ollama pull mistral)")
    print("   ✓ OR Gemini API key set? (export GEMINI_API_KEY=...)")
    print("=" * 60)
    print("\n🚀 Starting server...\n")
    
    web.run_app(app, access_log=None, host="0.0.0.0", port=8080)