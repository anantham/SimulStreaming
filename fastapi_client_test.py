import asyncio
import websockets
import sys
import argparse
import time

async def stream_audio(uri, audio_file, chunk_size=32000):
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        with open(audio_file, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                await websocket.send(data)
                # Simulate real-time streaming (assuming 16kHz mono 16-bit = 32000 bytes/sec)
                # If chunk_size is 32000, that's 1 second of audio.
                # Sleep a bit to not overwhelm server immediately (optional, server handles buffering)
                await asyncio.sleep(len(data) / 32000)
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    print(f"Received: {response}")
                except asyncio.TimeoutError:
                    pass
        
        # Keep listening for remaining responses
        print("Finished sending audio. Waiting for final responses...")
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"Received: {response}")
        except asyncio.TimeoutError:
            print("No more responses.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to 16kHz mono 16-bit PCM wav file")
    parser.add_argument("--uri", default="ws://localhost:43001/stream", help="WebSocket URI")
    args = parser.parse_args()
    
    try:
        asyncio.run(stream_audio(args.uri, args.audio_file))
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Error: {e}")
