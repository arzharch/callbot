import os
import aiohttp
import asyncio
from typing import List, Dict, Optional
import google.generativeai as genai
import json

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

class LLMProvider:
    """Unified interface for Ollama and Gemini with streaming."""
    
    def __init__(self, provider: str = None):
        self.provider = provider or LLM_PROVIDER
        self.session = None
        self.gemini_model = None
        
        if self.provider == "gemini" and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        print(f"🤖 LLM Provider: {self.provider.upper()}")
    
    async def initialize(self):
        """Initialize HTTP session for Ollama."""
        if self.provider == "ollama" and not self.session:
            self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def generate(self, messages: List[Dict], max_tokens: int = 150):
        """Generate response from LLM with streaming."""
        if self.provider == "gemini":
            async for chunk in self._generate_gemini(messages, max_tokens):
                yield chunk
        else:
            async for chunk in self._generate_ollama(messages, max_tokens):
                yield chunk
    
    async def _generate_ollama(self, messages: List[Dict], max_tokens: int):
        """Generate using Ollama with streaming."""
        if not self.session:
            await self.initialize()
        
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_tokens
                }
            }
            
            async with self.session.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_any():
                        try:
                            data = chunk.decode('utf-8')
                            for line in data.splitlines():
                                if line.strip():
                                    json_data = json.loads(line)
                                    if "message" in json_data and "content" in json_data['message']:
                                        yield json_data['message']['content']
                        except json.JSONDecodeError:
                            pass
                else:
                    error_text = await response.text()
                    print(f"Ollama API error: {response.status} - {error_text}")
                    yield "I'm having trouble thinking right now. Could you try again?"
        
        except asyncio.TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again."
        except aiohttp.ClientConnectorError:
            print("Cannot connect to Ollama")
            yield "I can't connect to my brain right now. Please check if Ollama is running."
        except Exception as e:
            print(f"Ollama error: {e}")
            yield "I encountered an error. Please try again."
    
    async def _generate_gemini(self, messages: List[Dict], max_tokens: int):
        """Generate using Gemini with streaming (fixed)."""
        if not self.gemini_model:
            yield "Gemini API not configured. Please set GEMINI_API_KEY."
            return
        
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            system_prompt = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt += msg["content"] + "\n"
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
            # Start chat with history
            chat = self.gemini_model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            
            # Get last user message
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            if system_prompt:
                last_message = f"{system_prompt}\n{last_message}"
            
            # Generate response with streaming (native sync)
            response_stream = chat.send_message(
                last_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                ),
                stream=True
            )
            
            # Stream chunks (sync iteration wrapped in executor)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                # Yield control back to event loop
                await asyncio.sleep(0)
        
        except Exception as e:
            print(f"Gemini error: {e}")
            yield "I encountered an error with Gemini. Please try again."

# Global instance
llm_provider = LLMProvider()