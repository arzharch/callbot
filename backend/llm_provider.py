import os
import aiohttp
import asyncio
from typing import List, Dict, Optional
import google.generativeai as genai

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "gemini"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

class LLMProvider:
    """Unified interface for Ollama and Gemini."""
    
    def __init__(self, provider: str = None):
        self.provider = provider or LLM_PROVIDER
        self.session = None
        self.gemini_model = None
        
        # Initialize Gemini if needed
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
    
    async def generate(self, messages: List[Dict], max_tokens: int = 150) -> str:
        """Generate response from LLM."""
        if self.provider == "gemini":
            return await self._generate_gemini(messages, max_tokens)
        else:
            return await self._generate_ollama(messages, max_tokens)
    
    async def _generate_ollama(self, messages: List[Dict], max_tokens: int) -> str:
        """Generate using Ollama."""
        if not self.session:
            await self.initialize()
        
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
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
                    result = await response.json()
                    return result['message']['content'].strip()
                else:
                    error_text = await response.text()
                    print(f"Ollama API error: {response.status} - {error_text}")
                    return "I'm having trouble thinking right now. Could you try again?"
        
        except asyncio.TimeoutError:
            return "Sorry, I'm taking too long to respond. Please try again."
        except aiohttp.ClientConnectorError:
            print("Cannot connect to Ollama")
            return "I can't connect to my brain right now. Please check if Ollama is running."
        except Exception as e:
            print(f"Ollama error: {e}")
            return "I encountered an error. Please try again."
    
    async def _generate_gemini(self, messages: List[Dict], max_tokens: int) -> str:
        """Generate using Gemini."""
        if not self.gemini_model:
            return "Gemini API not configured. Please set GEMINI_API_KEY."
        
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            system_prompt = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
            # Start chat with history
            chat = self.gemini_model.start_chat(history=gemini_messages[:-1])
            
            # Get last user message
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            
            # Generate response
            response = await asyncio.to_thread(
                chat.send_message,
                last_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            
            return response.text.strip()
        
        except Exception as e:
            print(f"Gemini error: {e}")
            return "I encountered an error with Gemini. Please try again."

# Global instance
llm_provider = LLMProvider()