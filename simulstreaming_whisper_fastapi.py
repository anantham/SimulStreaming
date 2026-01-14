#!/usr/bin/env python3
import sys
import argparse
import os
import logging
import numpy as np
import io
import soundfile
import librosa
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from simulstreaming_whisper import simulwhisper_args, simul_asr_factory
from whisper_streaming.whisper_online_main import processor_args, set_logging

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000

# Global ASR and OnlineProcessor objects (initialized in main)
asr_model = None
online_processor = None
min_chunk_seconds = 1.0

app = FastAPI()

class WebSocketProcessor:
    def __init__(self, websocket: WebSocket, online_asr_proc, min_chunk):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.is_first = True
        self.audio_buffer = []

    async def receive_audio_chunk(self):
        # Accumulate audio until min_chunk is reached or connection closes
        minlimit = self.min_chunk * SAMPLING_RATE
        
        while sum(len(x) for x in self.audio_buffer) < minlimit:
            try:
                # Receive raw bytes from WebSocket
                raw_bytes = await self.websocket.receive_bytes()
                
                if not raw_bytes:
                    break

                # Convert raw PCM16 bytes to float32 audio array
                sf = soundfile.SoundFile(
                    io.BytesIO(raw_bytes), 
                    channels=1, 
                    endian="LITTLE", 
                    samplerate=SAMPLING_RATE, 
                    subtype="PCM_16", 
                    format="RAW"
                )
                audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
                self.audio_buffer.append(audio)
                
            except WebSocketDisconnect:
                return None
            except Exception as e:
                logger.error(f"Error receiving audio: {e}")
                return None

        if not self.audio_buffer:
            return None

        conc = np.concatenate(self.audio_buffer)
        
        # If it's the first chunk, we might want to ensure we have enough data
        if self.is_first and len(conc) < minlimit:
             # Wait for more data in next iteration if possible, or just process what we have if stream ends?
             # For now, let's just return what we have if we break out of loop
             pass

        self.is_first = False
        
        # Clear buffer but keep remainder if any (simple implementation consumes all)
        self.audio_buffer = [] 
        return conc

    async def send_result(self, iteration_output):
        if iteration_output:
            # Format similar to the TCP server but sending JSON
            response = {
                'start': iteration_output['start'],
                'end': iteration_output['end'],
                'text': iteration_output['text']
            }
            try:
                await self.websocket.send_json(response)
                # Also log to console
                print(f"{response['start']:.0f} {response['end']:.0f} {response['text']}", flush=True)
            except WebSocketDisconnect:
                pass
        else:
            logger.debug("No text in this segment")

    async def process(self):
        # Initialize the processor for this session
        # Note: simul_whisper might strictly need re-initialization per session.
        # The online_processor object might be stateful. 
        # Ideally, we should create a NEW processor per connection or reset it.
        # Check SimulWhisperOnline.init()
        self.online_asr_proc.init()
        
        try:
            while True:
                a = await self.receive_audio_chunk()
                if a is None:
                    break
                
                self.online_asr_proc.insert_audio_chunk(a)
                o = self.online_asr_proc.process_iter()
                
                await self.send_result(o)
                
                # Small yield to allow other tasks
                await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    # Create a processor wrapper for this connection
    # We use the global online_processor. 
    # WARNING: If SimulWhisperOnline is not thread-safe or supports concurrent sessions, 
    # this global usage will fail for multiple clients.
    # The original server was single-threaded/sequential (accept, process, close, accept next).
    # To support multiple clients, we would need to instantiate a new ASR/Processor per client 
    # OR lock the single instance.
    # For now, assuming single active client or lightweight re-init.
    # But `simul_asr_factory` loads the model which is heavy. 
    # We should reuse the MODEL but create new ONLINE PROCESSOR.
    
    # Re-creating the online processor wrapper for the existing ASR model:
    from simulstreaming_whisper import SimulWhisperOnline
    session_processor = SimulWhisperOnline(asr_model)
    
    processor = WebSocketProcessor(websocket, session_processor, min_chunk_seconds)
    await processor.process()


@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>SimulStreaming Whisper</title>
        </head>
        <body>
            <h1>SimulStreaming Whisper WebSocket</h1>
            <p>Connect to <pre>ws://host:port/stream</pre> and send raw PCM16 16kHz audio bytes.</p>
        </body>
    </html>
    """)

def main():
    global asr_model, min_chunk_seconds
    
    parser = argparse.ArgumentParser()
    
    # Server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=43001)
    
    # Whisper/SimulStreaming options
    processor_args(parser)
    simulwhisper_args(parser)
    
    args = parser.parse_args()
    set_logging(args, logger)
    
    # Initialize ASR
    # factory returns (asr, online). We keep 'asr' to create new 'online' instances per client.
    asr_model, _ = simul_asr_factory(args)
    
    if args.vac:
        min_chunk_seconds = args.vac_chunk_size
    else:
        min_chunk_seconds = args.min_chunk_size

    # Warmup
    if not args.model_path or not os.path.exists(args.model_path):
        logger.info("Model path might be downloaded or invalid. Checking...")

    # Run FastAPI
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
