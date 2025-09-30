
# AI WebRTC Call Bot

This project is a Python-based WebRTC chat application that uses an AI model (Ollama with Mistral) to have a free-flowing conversation with a user. It's designed to be the foundation for a voice-based callbot.

## Features

- **Asynchronous Backend:** Built with `aiohttp` for high performance.
- **WebRTC Communication:** Uses `aiortc` for real-time, low-latency communication.
- **LLM Integration:** Connects to an Ollama instance to generate intelligent responses.
- **Streaming Responses:** Bot responses are streamed token-by-token for a real-time feel.
- **Tool Calling:** The bot can decide to use tools to perform actions like querying a knowledge base or booking tickets.
- **Persistent Memory:** Uses Redis to store conversation history for each user (keyed by phone number).
- **Clean Frontend/Backend Separation:** The project is organized into distinct `frontend` and `backend` directories.

## Prerequisites

Before you begin, you need to have the following installed and running:

1.  **Python 3.9+**
2.  **Redis:** A Redis server running on `localhost:6379`.
3.  **Ollama:** An Ollama instance running and serving the `mistral` model. The application expects the API to be at `http://localhost:11434`.
    - You can pull the model with: `ollama pull mistral`

## Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd callbot/backend
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Ensure your prerequisites (Redis, Ollama) are running.**

2.  **Navigate to the `backend` directory** (if you aren't already there) and run the main application:
    ```bash
    python main.py
    ```

3.  You should see output indicating that the server is running on `http://0.0.0.0:8080` and has successfully connected to Redis.

4.  **Open your web browser** and navigate to:
    [http://localhost:8080](http://localhost:8080)

5.  The chat interface will load. The bot will ask for your phone number, and once you provide it, you can start the conversation.

## How It Works

- The `aiohttp` server in `backend/main.py` serves the `frontend/index.html` and handles the WebRTC signaling.
- When a user connects, a `RTCPeerConnection` is established, and a `ConversationManager` instance from `backend/bot_logic.py` is created for that user.
- User messages are sent over the WebRTC data channel.
- The `ConversationManager` handles the state (e.g., waiting for a phone number).
- It constructs a prompt with history and tool definitions and sends it to the Ollama API.
- If the LLM decides to use a tool, the backend executes the corresponding function from `backend/tools.py`.
- The result is sent back to the LLM to generate a final, user-friendly response.
- The final response is streamed back to the user over the data channel.
