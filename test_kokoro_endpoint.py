"""Test script to check Kokoro FastAPI endpoint format."""

import asyncio
import aiohttp


async def test_kokoro_endpoint():
    """Test the Kokoro FastAPI endpoint to see what format it returns."""
    
    base_url = "http://localhost:8880"
    endpoint = "/v1/audio/speech"
    
    payload = {
        "input": "Hello, this is a test.",
        "voice": "af_sarah",
        "speed": 1.0,
        "model": "kokoro",
    }
    
    print(f"Testing: {base_url}{endpoint}")
    print(f"Payload: {payload}\n")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}\n")
            
            if response.status == 200:
                audio_bytes = await response.read()
                print(f"Response size: {len(audio_bytes)} bytes")
                
                # Check first few bytes
                print(f"First 16 bytes (hex): {audio_bytes[:16].hex()}")
                print(f"First 4 bytes (text): {audio_bytes[:4]}")
                
                # Check if it's a WAV file
                if audio_bytes.startswith(b'RIFF'):
                    print("✓ Format: WAV file")
                    # Parse WAV header
                    import struct
                    chunk_id = audio_bytes[0:4]
                    chunk_size = struct.unpack('<I', audio_bytes[4:8])[0]
                    format_tag = audio_bytes[8:12]
                    print(f"  ChunkID: {chunk_id}")
                    print(f"  ChunkSize: {chunk_size}")
                    print(f"  Format: {format_tag}")
                else:
                    print("✓ Format: Raw audio data (likely float32 PCM)")
                    # Try to interpret as float32
                    import struct
                    num_samples = min(4, len(audio_bytes) // 4)
                    samples = struct.unpack(f'{num_samples}f', audio_bytes[:num_samples*4])
                    print(f"  First {num_samples} float32 samples: {samples}")
            else:
                error_text = await response.text()
                print(f"Error: {error_text}")


if __name__ == "__main__":
    asyncio.run(test_kokoro_endpoint())