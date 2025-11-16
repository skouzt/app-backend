"""
Complete Bot Pipeline with Whisper STT + Gemini 2.5 Flash + Kokoro TTS
Includes Vapi-style WebSocket message broadcasting
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame,
    EndFrame,
)

# Daily transport for 0.0.82
from pipecat.transports.services.daily import DailyTransport, DailyParams

# VAD for 0.0.82

from services.whisper_stt import create_whisper_service
from services.gemini_llm import GeminiLLMService
from services.kokoro_tts import KokoroTTSService

logger = logging.getLogger(__name__)


class VapiMessageProcessor(FrameProcessor):
    """
    Processor that sends Vapi-style messages to WebSocket
    Converts Pipecat frames to Vapi message format
    """
    
    def __init__(self, call_id: str, broadcast_callback):
        super().__init__()
        self.call_id = call_id
        self.broadcast = broadcast_callback
        self._user_speaking = False
        self._assistant_speaking = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Convert frames to Vapi messages"""
        
        try:
            # User transcript (STT output)
            if isinstance(frame, TranscriptionFrame):
                if frame.user_id == "user":
                    is_final = getattr(frame, 'is_final', True)
                    
                    # Send transcript message
                    await self.broadcast(self.call_id, {
                        "type": "transcript",
                        "role": "user",
                        "transcriptType": "final" if is_final else "partial",
                        "transcript": frame.text,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Send speech update
                    if is_final and self._user_speaking:
                        self._user_speaking = False
                        await self.broadcast(self.call_id, {
                            "type": "speech-update",
                            "role": "user",
                            "status": "stopped",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    elif not is_final and not self._user_speaking:
                        self._user_speaking = True
                        await self.broadcast(self.call_id, {
                            "type": "speech-update",
                            "role": "user",
                            "status": "started",
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            # Assistant transcript (LLM output)
            elif isinstance(frame, TextFrame):
                # Send transcript
                await self.broadcast(self.call_id, {
                    "type": "transcript",
                    "role": "assistant",
                    "transcriptType": "final",
                    "transcript": frame.text,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send conversation update
                await self.broadcast(self.call_id, {
                    "type": "conversation-update",
                    "message": {
                        "role": "assistant",
                        "content": frame.text
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Speech update
                if not self._assistant_speaking:
                    self._assistant_speaking = True
                    await self.broadcast(self.call_id, {
                        "type": "speech-update",
                        "role": "assistant",
                        "status": "started",
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        except Exception as e:
            logger.error(f"Error in VapiMessageProcessor: {e}", exc_info=True)
        
        # Always push frame forward
        await self.push_frame(frame, direction)


async def create_bot_pipeline(
    call_id: str,
    room_url: str,
    bot_token: str,
    assistant_config: dict,
    broadcast_callback,
    gemini_api_key: str,
    whisper_model: str = "base",
    kokoro_voice: str = "af_sarah",
    use_faster_whisper: bool = True
):
    """
    Create complete bot pipeline with all services
    
    Args:
        call_id: Unique call identifier
        room_url: Daily.co room URL
        bot_token: Daily.co bot token
        assistant_config: Assistant configuration
        broadcast_callback: Function to broadcast WebSocket messages
        gemini_api_key: Google Gemini API key
        whisper_model: Whisper model size
        kokoro_voice: Kokoro voice name
        use_faster_whisper: Use faster-whisper if available
    
    Returns:
        PipelineTask ready to run
    """
    
    logger.info(f"Creating bot pipeline for call: {call_id}")
    
    # 1. Initialize Daily transport
    transport = DailyTransport(
        room_url,
        bot_token,
        assistant_config.get("name", "Dr. Sarah"),
       DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=False,
                vad_analyzer=None,
                transcription_enabled=False,
            )

    )
    
    # 2. Initialize Whisper STT
    stt = create_whisper_service(
        model_size=whisper_model,
        device="cpu",  # Change to "cuda" if GPU available
        language="en",
        use_faster=use_faster_whisper
    )
    
    # 3. Initialize Gemini 2.5 Flash LLM
    llm = GeminiLLMService(
        api_key=gemini_api_key,
        model=assistant_config.get("model", "gemini-2.0-flash-exp"),
        temperature=0.7,
        max_output_tokens=2048
    )
    
    # 4. Initialize Kokoro TTS
    tts = KokoroTTSService(
        voice=kokoro_voice,
        device="cpu",  # Change to "cuda" if GPU available
        speed=1.0
    )
    
    # 5. Create message aggregators
    user_response = LLMUserResponseAggregator()
    assistant_response = LLMAssistantResponseAggregator()
    
    # 6. Create Vapi message processor
    vapi_processor = VapiMessageProcessor(call_id, broadcast_callback)
    
    # 7. Build pipeline
    pipeline = Pipeline([
        transport.input(),      # Audio from user
        stt,                    # Speech to text (Whisper)
        user_response,          # Aggregate user messages
        vapi_processor,         # Send WebSocket messages
        llm,                    # Generate response (Gemini)
        tts,                    # Text to speech (Kokoro)
        assistant_response,     # Aggregate assistant messages
        transport.output()      # Audio to user
    ])
    
    # 8. Create task
    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    
    # 9. Set initial context with system prompt
    system_prompt = assistant_config.get("systemPrompt", "You are a helpful assistant.")
    first_message = assistant_config.get("firstMessage", "Hello! How can I help you?")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": first_message}
    ]
    
    await task.queue_frames([llm.create_context_frame(messages)])
    
    # 10. Send initial greeting via WebSocket
    await broadcast_callback(call_id, {
        "type": "transcript",
        "role": "assistant",
        "transcriptType": "final",
        "transcript": first_message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    await broadcast_callback(call_id, {
        "type": "conversation-update",
        "message": {
            "role": "assistant",
            "content": first_message
        },
        "timestamp": datetime.utcnow().isoformat()
    })
    
    logger.info(f"Bot pipeline created successfully for call: {call_id}")
    
    return task


async def run_bot_pipeline(
    call_id: str,
    room_url: str,
    bot_token: str,
    assistant_config: dict,
    broadcast_callback,
    gemini_api_key: str,
    whisper_model: str = "base",
    kokoro_voice: str = "af_sarah",
    use_faster_whisper: bool = True
):
    """
    Run the complete bot pipeline
    
    This is the main function to start a bot for a call
    """
    try:
        logger.info(f"Starting bot pipeline for call: {call_id}")
        
        # Create pipeline
        task = await create_bot_pipeline(
            call_id=call_id,
            room_url=room_url,
            bot_token=bot_token,
            assistant_config=assistant_config,
            broadcast_callback=broadcast_callback,
            gemini_api_key=gemini_api_key,
            whisper_model=whisper_model,
            kokoro_voice=kokoro_voice,
            use_faster_whisper=use_faster_whisper
        )
        
        # Run pipeline
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info(f"Bot pipeline finished for call: {call_id}")
        
    except Exception as e:
        logger.error(f"Error in bot pipeline for call {call_id}: {e}", exc_info=True)
        
        # Send error message via WebSocket
        await broadcast_callback(call_id, {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        raise


# Example usage for testing
"""
async def main():
    # Assistant configuration
    assistant_config = {
        "name": "Dr. Sarah",
        "model": "gemini-2.0-flash-exp",
        "voice": "af_sarah",
        "systemPrompt": '''You are Dr. Sarah, a compassionate AI therapist.
        
Core principles:
- Practice empathetic listening
- Ask open-ended questions
- Keep responses concise (2-4 sentences)
- Never diagnose or prescribe
- Recommend professional help in crisis

Be warm, caring, and supportive.''',
        "firstMessage": "Hello! I'm Dr. Sarah. I'm here to listen and support you. How are you feeling today?"
    }
    
    # Mock broadcast function
    async def broadcast(call_id, message):
        print(f"[{call_id}] {message['type']}: {message.get('transcript', message)}")
    
    # Run pipeline
    await run_bot_pipeline(
        call_id="test_call_123",
        room_url="https://company.daily.co/test-room",
        bot_token="test_token",
        assistant_config=assistant_config,
        broadcast_callback=broadcast,
        gemini_api_key="your-gemini-api-key",
        whisper_model="base",
        kokoro_voice="af_sarah",
        use_faster_whisper=True
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""