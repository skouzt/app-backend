"""
AI Therapist Backend - Vapi-Compatible API (No Auth)
Simple version with Whisper STT + OpenAI LLM + Kokoro TTS
"""

import os
import asyncio
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from bot_pipeline import run_bot_pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate required environment variables
REQUIRED_ENV_VARS = ["GEMINI_API_KEY", "DAILY_API_KEY"]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

# Store active calls and WebSocket connections
active_calls: Dict[str, dict] = {}
websocket_connections: Dict[str, list] = {}

# Default assistant config
DEFAULT_ASSISTANT = {
    "name": "Dr. Sarah",
    "voice": "af_sarah",
    "model": "gpt-4o-mini",
    "firstMessage": "Hello, I'm Dr. Sarah. I'm here to listen and support you. How are you feeling today?",
    "systemPrompt": """You are Dr. Sarah, a compassionate AI therapist with expertise in CBT and active listening.

Core principles:
- Practice empathetic listening and validate emotions
- Ask open-ended questions
- Use evidence-based therapeutic techniques
- Keep responses concise (2-4 sentences)
- Never diagnose or prescribe medication
- In crisis, recommend professional help

Be warm, caring, and supportive."""
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    logger.info("Starting AI Therapist Backend (Whisper + OpenAI + Kokoro)...")
    yield
    logger.info("Shutting down...")
    for call_id in list(active_calls.keys()):
        try:
            await cleanup_call(call_id)
        except Exception as e:
            logger.error(f"Error cleaning up call {call_id}: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="AI Therapist API (Vapi-Compatible)",
    description="Whisper STT + OpenAI LLM + Kokoro TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class Assistant(BaseModel):
    """Assistant configuration"""
    name: str = "Dr. Sarah"
    voice: str = "af_sarah"  # Kokoro voice
    model: str = "gpt-4o-mini"
    firstMessage: str = DEFAULT_ASSISTANT["firstMessage"]
    systemPrompt: str = DEFAULT_ASSISTANT["systemPrompt"]

class CallRequest(BaseModel):
    """Create call request"""
    assistant: Optional[Assistant] = None

class CallResponse(BaseModel):
    """Call response"""
    id: str
    createdAt: str
    updatedAt: str
    status: str
    type: str = "webCall"
    assistant: Assistant
    webCallUrl: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Therapist API",
        "version": "1.0.0",
        "status": "running",
        "features": {
            "stt": "whisper",
            "llm": "openai-gpt4o-mini",
            "tts": "kokoro-82m"
        },
        "endpoints": {
            "health": "GET /health",
            "call": {
                "create": "POST /call",
                "get": "GET /call/{call_id}",
                "delete": "DELETE /call/{call_id}",
                "list": "GET /call"
            },
            "websocket": "WS /call/{call_id}/ws"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "activeCalls": len(active_calls),
        "services": {
            "stt": "whisper",
            "llm": "openai",
            "tts": "kokoro-82m"
        }
    }

@app.post("/call", response_model=CallResponse)
async def create_call(request: CallRequest):
    """
    Create a new call (Vapi-style)
    No authentication required
    """
    try:
        logger.info("Creating new call")
        
        # Generate call ID
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        
        # Use provided assistant or default
        assistant = request.assistant or Assistant(**DEFAULT_ASSISTANT)
        
        # Calculate expiration (1 hour)
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(hours=1)
        
        # Create Daily.co room
        async with httpx.AsyncClient() as client:
            daily_response = await client.post(
                "https://api.daily.co/v1/rooms",
                headers={
                    "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "properties": {
                        "exp": int(expires_at.timestamp()),
                        "max_participants": 2,
                        "enable_chat": False,
                        "enable_screenshare": False,
                        "start_video_off": True,
                        "start_audio_off": False,
                        "eject_at_room_exp": True
                    }
                },
                timeout=10.0
            )
            
            if daily_response.status_code != 200:
                logger.error(f"Daily API error: {daily_response.text}")
                raise HTTPException(status_code=500, detail="Failed to create room")
            
            room_data = daily_response.json()
        
        room_url = room_data["url"]
        room_name = room_data["name"]
        
        # Create bot token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://api.daily.co/v1/meeting-tokens",
                headers={
                    "Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "properties": {
                        "room_name": room_name,
                        "is_owner": True
                    }
                },
                timeout=10.0
            )
            
            if token_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to create token")
            
            bot_token = token_response.json()["token"]
        
        # Store call info
        call_info = {
            "id": call_id,
            "roomUrl": room_url,
            "roomName": room_name,
            "botToken": bot_token,
            "assistant": assistant.dict(),
            "status": "queued",
            "type": "webCall",
            "createdAt": created_at.isoformat(),
            "updatedAt": created_at.isoformat(),
            "expiresAt": expires_at.isoformat(),
        }
        
        active_calls[call_id] = call_info
        websocket_connections[call_id] = []
        
        # Start bot in background
        asyncio.create_task(start_bot(call_id, room_url, bot_token, assistant.dict()))
        
        logger.info(f"Call created: {call_id}")
        
        return CallResponse(
            id=call_id,
            createdAt=created_at.isoformat(),
            updatedAt=created_at.isoformat(),
            status="queued",
            type="webCall",
            assistant=assistant,
            webCallUrl=room_url
        )
        
    except Exception as e:
        logger.error(f"Error creating call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/call/{call_id}", response_model=CallResponse)
async def get_call(call_id: str):
    """Get call details"""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call_data = active_calls[call_id]
    
    return CallResponse(
        id=call_data["id"],
        createdAt=call_data["createdAt"],
        updatedAt=call_data["updatedAt"],
        type=call_data["type"],
        status=call_data["status"],
        assistant=Assistant(**call_data["assistant"]),
        webCallUrl=call_data["roomUrl"]
    )

@app.delete("/call/{call_id}")
async def delete_call(call_id: str):
    """End/delete call"""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
    
    await cleanup_call(call_id)
    
    return {
        "id": call_id,
        "status": "ended",
        "endedAt": datetime.utcnow().isoformat()
    }

@app.get("/call")
async def list_calls():
    """List all active calls"""
    calls = [
        {
            "id": call_id,
            "status": data["status"],
            "createdAt": data["createdAt"],
            "type": data["type"]
        }
        for call_id, data in active_calls.items()
    ]
    return {"calls": calls}

# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/call/{call_id}/ws")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time conversation (Vapi-style)
    """
    try:
        # Verify call exists
        if call_id not in active_calls:
            await websocket.close(code=1008, reason="Call not found")
            return
        
        # Accept connection
        await websocket.accept()
        logger.info(f"WebSocket connected for call: {call_id}")
        
        # Store connection
        websocket_connections[call_id].append(websocket)
        
        # Send connection message
        await websocket.send_json({
            "type": "connection",
            "callId": call_id,
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update call status
        active_calls[call_id]["status"] = "in-progress"
        
        try:
            # Keep connection alive
            while True:
                data = await websocket.receive_json()
                await handle_client_message(websocket, call_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for call: {call_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Remove connection
            if websocket in websocket_connections.get(call_id, []):
                websocket_connections[call_id].remove(websocket)
            
            # Update status if no more connections
            if not websocket_connections.get(call_id):
                active_calls[call_id]["status"] = "ended"
    
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal error")
        except:
            pass

async def handle_client_message(websocket: WebSocket, call_id: str, message: dict):
    """Handle incoming messages from client"""
    message_type = message.get("type")
    
    if message_type == "control":
        action = message.get("action")
        if action == "interrupt":
            await broadcast_to_call(call_id, {
                "type": "speech-update",
                "role": "assistant",
                "status": "stopped"
            })

async def broadcast_to_call(call_id: str, message: dict):
    """Broadcast message to all WebSocket connections for a call"""
    connections = websocket_connections.get(call_id, [])
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting: {e}")

# ============================================================================
# BOT MANAGEMENT
# ============================================================================

async def start_bot(call_id: str, room_url: str, bot_token: str, assistant_config: dict):
    """Start the bot pipeline"""
    try:
        logger.info(f"Starting bot for call: {call_id}")
        
        # Update status
        if call_id in active_calls:
            active_calls[call_id]["status"] = "ringing"
        
        # Get configuration
        whisper_model = os.getenv("WHISPER_MODEL", "base")
        kokoro_voice = assistant_config.get("voice", "af_sarah")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # Run bot pipeline
        await run_bot_pipeline(
            call_id=call_id,
            room_url=room_url,
            bot_token=bot_token,
            assistant_config=assistant_config,
            broadcast_callback=broadcast_to_call,
            openai_api_key=openai_key,
            whisper_model=whisper_model,
            kokoro_voice=kokoro_voice
        )
        
    except Exception as e:
        logger.error(f"Error in bot: {e}", exc_info=True)
        if call_id in active_calls:
            active_calls[call_id]["status"] = "error"

async def cleanup_call(call_id: str):
    """Cleanup call resources"""
    if call_id in active_calls:
        call_data = active_calls[call_id]
        
        # Close WebSocket connections
        for ws in websocket_connections.get(call_id, []):
            try:
                await ws.close()
            except:
                pass
        
        # Delete Daily room
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"https://api.daily.co/v1/rooms/{call_data['roomName']}",
                    headers={"Authorization": f"Bearer {os.getenv('DAILY_API_KEY')}"},
                    timeout=10.0
                )
        except Exception as e:
            logger.error(f"Error deleting Daily room: {e}")
        
        # Remove from storage
        del active_calls[call_id]
        if call_id in websocket_connections:
            del websocket_connections[call_id]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=True)