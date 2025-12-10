# voice_assistant_sounddevice_final.py
from __future__ import annotations
import os
import sys
import argparse
import asyncio
import base64
from datetime import datetime
import logging
import queue
import signal
from dotenv import load_dotenv

import sounddevice as sd
import numpy as np

from typing import Union, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential, DefaultAzureCredential

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AzureStandardVoice,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad
)

# Load environment variables
load_dotenv(override=True)

# Logging
if not os.path.exists("logs"):
    os.makedirs("logs")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfilename = f"{timestamp}_conversation.log"

logging.basicConfig(
    filename=f"logs/{timestamp}_voicelive.log",
    filemode="w",
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------------
# Device selection utilities
# ------------------------------
def select_device_and_rate(preferred_rate: int = 24000, channels: int = 1):
    """
    Auto-detect input and output devices and a working sample rate.
    Tries preferred_rate first; falls back to 16000 if needed.
    Returns: (input_device_id, output_device_id, chosen_rate)
    """
    devices = sd.query_devices()
    default_input, default_output = sd.default.device

    # Helper to find first device with required channels
    def find_device(kind: str):
        # kind: "input" or "output"
        for i, d in enumerate(devices):
            if kind == "input":
                if d.get("max_input_channels", 0) >= channels:
                    return i, d
            else:
                if d.get("max_output_channels", 0) >= channels:
                    return i, d
        return None, None

    # Try using current defaults first
    chosen_rate = preferred_rate
    input_device = None
    output_device = None

    # Try preferred rate with defaults
    try:
        sd.check_input_settings(device=default_input, samplerate=preferred_rate, channels=channels)
        sd.check_output_settings(device=default_output, samplerate=preferred_rate, channels=channels)
        input_device = default_input
        output_device = default_output
        logger.info("Using system default devices at %d Hz", preferred_rate)
        return input_device, output_device, chosen_rate
    except Exception:
        logger.debug("Default devices do not support %d Hz, searching alternatives...", preferred_rate)

    # Try to locate devices that support preferred_rate
    in_dev_idx, in_dev_info = find_device("input")
    out_dev_idx, out_dev_info = find_device("output")

    # If devices found, try preferred rate; if not supported, use device's default_samplerate
    def try_device_rate(dev_idx, dev_info, rate):
        if dev_idx is None or dev_info is None:
            return False
        try:
            sd.check_input_settings(device=dev_idx, samplerate=rate, channels=channels) if dev_info.get("max_input_channels", 0) >= channels else None
            sd.check_output_settings(device=dev_idx, samplerate=rate, channels=channels) if dev_info.get("max_output_channels", 0) >= channels else None
            return True
        except Exception:
            return False

    # Prefer devices that explicitly support preferred_rate (or have default_samplerate >= preferred_rate)
    if in_dev_idx is not None:
        if try_device_rate(in_dev_idx, in_dev_info, preferred_rate):
            input_device = in_dev_idx
    if out_dev_idx is not None:
        if try_device_rate(out_dev_idx, out_dev_info, preferred_rate):
            output_device = out_dev_idx

    # If either device still None, iterate devices to find any that can do preferred_rate
    if input_device is None:
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) >= channels:
                try:
                    sd.check_input_settings(device=i, samplerate=preferred_rate, channels=channels)
                    input_device = i
                    break
                except Exception:
                    continue

    if output_device is None:
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) >= channels:
                try:
                    sd.check_output_settings(device=i, samplerate=preferred_rate, channels=channels)
                    output_device = i
                    break
                except Exception:
                    continue

    # If still no devices supporting preferred_rate, try fallback rate 16000
    if input_device is None or output_device is None:
        fallback_rate = 16000
        logger.info("Falling back to %d Hz because %d Hz was not available", fallback_rate, preferred_rate)
        chosen_rate = fallback_rate

        # try defaults first for fallback
        try:
            sd.check_input_settings(device=default_input, samplerate=chosen_rate, channels=channels)
            input_device = input_device or default_input
        except Exception:
            # find any input device that accepts fallback rate
            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) >= channels:
                    try:
                        sd.check_input_settings(device=i, samplerate=chosen_rate, channels=channels)
                        input_device = i
                        break
                    except Exception:
                        continue

        try:
            sd.check_output_settings(device=default_output, samplerate=chosen_rate, channels=channels)
            output_device = output_device or default_output
        except Exception:
            for i, d in enumerate(devices):
                if d.get("max_output_channels", 0) >= channels:
                    try:
                        sd.check_output_settings(device=i, samplerate=chosen_rate, channels=channels)
                        output_device = i
                        break
                    except Exception:
                        continue

    # Last resort: if still None, pick first available input/output device (may still cause errors)
    if input_device is None:
        idx, _ = find_device("input")
        input_device = idx
    if output_device is None:
        idx, _ = find_device("output")
        output_device = idx

    logger.info("Selected input device: %s, output device: %s, rate: %s", input_device, output_device, chosen_rate)
    return input_device, output_device, chosen_rate


# ------------------------------
# AudioProcessor using sounddevice
# ------------------------------
class AudioProcessor:
    """
    Handles real-time audio capture and playback using sounddevice.
    Auto-detects devices and sample rate.
    """

    class AudioPlaybackPacket:
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection):
        self.connection = connection

        # default configuration - may be overridden by device selection
        self.preferred_rate = 24000
        self.channels = 1
        self.dtype = "int16"
        self.chunk_size = 1200  # 50 ms at 24kHz

        # Auto-select devices and effective rate
        input_device, output_device, effective_rate = select_device_and_rate(preferred_rate=self.preferred_rate, channels=self.channels)
        self.input_device = input_device
        self.output_device = output_device
        self.rate = effective_rate

        # adjust chunk_size if rate changed (keeps ~50ms)
        self.chunk_size = int(round(0.05 * self.rate))

        logger.info("AudioProcessor init: input_device=%s output_device=%s rate=%d chunk_size=%d", self.input_device, self.output_device, self.rate, self.chunk_size)
        print(f"🎤 Using input device: {self.input_device}")
        print(f"🔈 Using output device: {self.output_device}")
        print(f"⚡ Sample rate: {self.rate} Hz, chunk size: {self.chunk_size} frames")

        # Streams
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None

        # Playback queue
        self.playback_queue: "queue.Queue[AudioProcessor.AudioPlaybackPacket]" = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0

        # event loop placeholder (populated when capture starts)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def start_capture(self):
        """Start microphone capture and push base64 PCM16 chunks to connection.input_audio_buffer"""
        if self.input_stream:
            return

        self.loop = asyncio.get_event_loop()

        def callback(indata, frames, time, status):
            if status:
                logger.debug("Input status: %s", status)

            # indata is a NumPy array, convert to bytes (PCM16)
            try:
                raw_bytes = indata.tobytes()
                audio_b64 = base64.b64encode(raw_bytes).decode("utf-8")
                # Append to the voice live input buffer asynchronously
                asyncio.run_coroutine_threadsafe(
                    self.connection.input_audio_buffer.append(audio=audio_b64),
                    self.loop
                )
            except Exception as e:
                logger.exception("Exception in capture callback: %s", e)

        self.input_stream = sd.InputStream(
            device=self.input_device,
            channels=self.channels,
            samplerate=self.rate,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            callback=callback,
        )

        try:
            self.input_stream.start()
            logger.info("Microphone capture started.")
        except Exception:
            logger.exception("Failed to start microphone capture")
            raise

    def start_playback(self):
        """Start output stream and consume queued PCM16 bytes."""
        if self.output_stream:
            return

        # We'll keep a small leftover buffer between callbacks
        leftover = bytearray()

        def callback(outdata, frames, time, status):
            nonlocal leftover
            if status:
                logger.debug("Output status: %s", status)

            needed_bytes = frames * self.channels * 2  # int16 -> 2 bytes
            out_bytes = bytearray()

            # first use leftover
            if leftover:
                take = min(len(leftover), needed_bytes)
                out_bytes += leftover[:take]
                leftover = leftover[take:]

            while len(out_bytes) < needed_bytes:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    # fill remaining with silence
                    out_bytes += b"\x00" * (needed_bytes - len(out_bytes))
                    break

                if packet is None or packet.data is None:
                    # end of stream marker -> fill silence
                    out_bytes += b"\x00" * (needed_bytes - len(out_bytes))
                    break

                # Skip old packets
                if packet.seq_num < self.playback_base:
                    continue

                need = needed_bytes - len(out_bytes)
                take = packet.data[:need]
                out_bytes += take
                if len(packet.data) > len(take):
                    # store leftover for next callback
                    leftover = bytearray(packet.data[len(take):])

            # Convert bytes to numpy array and write to outdata
            try:
                arr = np.frombuffer(bytes(out_bytes), dtype=np.int16)
                # reshape to (frames, channels)
                if arr.size != frames * self.channels:
                    # if mismatch, pad/truncate
                    arr = np.resize(arr, frames * self.channels)
                arr = arr.reshape((frames, self.channels))
                outdata[:] = arr
            except Exception as e:
                logger.exception("Playback conversion error: %s", e)
                # fill with silence on error
                outdata[:] = np.zeros((frames, self.channels), dtype=np.int16)

        self.output_stream = sd.OutputStream(
            device=self.output_device,
            channels=self.channels,
            samplerate=self.rate,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            callback=callback,
        )

        try:
            self.output_stream.start()
            logger.info("Audio playback started.")
        except Exception:
            logger.exception("Failed to start audio playback")
            raise

    def queue_audio(self, audio_data: Optional[bytes]):
        """Queue PCM16 bytes for playback."""
        pkt = AudioProcessor.AudioPlaybackPacket(seq_num=self.next_seq_num, data=audio_data)
        self.next_seq_num += 1
        self.playback_queue.put(pkt)

    def skip_pending_audio(self):
        """Skip audio currently queued (used on barge-in)."""
        self.playback_base = self.next_seq_num

    def shutdown(self):
        """Clean up streams."""
        try:
            if self.input_stream:
                try:
                    self.input_stream.stop()
                    self.input_stream.close()
                except Exception:
                    logger.debug("Error stopping input stream", exc_info=True)
                self.input_stream = None

            if self.output_stream:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                except Exception:
                    logger.debug("Error stopping output stream", exc_info=True)
                self.output_stream = None

            logger.info("AudioProcessor shutdown complete.")
        except Exception as e:
            logger.exception("Exception during shutdown: %s", e)


# ------------------------------
# Main assistant class (unchanged behavior)
# ------------------------------
class BasicVoiceAssistant:
    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, AsyncTokenCredential],
        agent_id: str,
        foundry_project_name: str,
        voice: str,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.agent_id = agent_id
        self.foundry_project_name = foundry_project_name
        self.voice = voice

        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None

        self.session_ready = False
        self.conversation_started = False
        self._active_response = False
        self._response_api_done = False

    async def start(self):
        try:
            logger.info("Connecting to VoiceLive API with Foundry agent %s", self.agent_id)

            # Obtain token using DefaultAzureCredential (async)
            token = (await DefaultAzureCredential().get_token("https://ai.azure.com/.default")).token

            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                query={
                    "agent-id": self.agent_id,
                    "agent-project-name": self.foundry_project_name,
                    "agent-access-token": token,
                },
            ) as connection:
                conn = connection
                self.connection = conn

                # Init audio processor with connection
                ap = AudioProcessor(conn)
                self.audio_processor = ap

                # Configure session & start playback
                await self._setup_session()
                ap.start_playback()

                print("\n" + "=" * 60)
                print("🎤 VOICE ASSISTANT READY")
                print("Start speaking to begin conversation")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                # Process events
                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self):
        logger.info("Setting up voice session...")

        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            voice_config = AzureStandardVoice(name=self.voice, rate="0.90")
        else:
            voice_config = self.voice

        turn_detection_config = ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)

        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=AudioEchoCancellation(),
            input_audio_noise_reduction=AudioNoiseReduction(type="azure_deep_noise_suppression"),
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)
        logger.info("Session configuration sent")

    async def _process_events(self):
        try:
            conn = self.connection
            assert conn is not None
            async for event in conn:
                await self._handle_event(event)
        except Exception:
            logger.exception("Error processing events")
            raise

    async def _handle_event(self, event):
        logger.debug("Received event: %s", event.type)
        ap = self.audio_processor
        conn = self.connection
        assert ap is not None
        assert conn is not None

        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info("Session ready: %s", event.session.id)
            await write_conversation_log(f"SessionID: {event.session.id}")
            await write_conversation_log(f"Model: {event.session.model}")
            await write_conversation_log(f"Voice: {event.session.voice}")
            await write_conversation_log(f"Instructions: {event.session.instructions}")
            await write_conversation_log("")
            self.session_ready = True

            if not self.conversation_started:
                self.conversation_started = True
                logger.info("Sending proactive greeting request")
                try:
                    await conn.response.create()
                except Exception:
                    logger.exception("Failed to send proactive greeting request")

            ap.start_capture()

        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            print(f"👤 You said:\t{event.get('transcript', '')}")
            await write_conversation_log(f"User Input:\t{event.get('transcript', '')}")

        elif event.type == ServerEventType.RESPONSE_TEXT_DONE:
            print(f"🤖 Agent responded with text:\t{event.get('text', '')}")
            await write_conversation_log(f"Agent Text Response:\t{event.get('text', '')}")

        elif event.type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
            print(f"🤖 Agent responded with audio transcript:\t{event.get('transcript', '')}")
            await write_conversation_log(f"Agent Audio Response:\t{event.get('transcript', '')}")

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("User started speaking - stopping playback")
            print("🎤 Listening...")
            ap.skip_pending_audio()
            if self._active_response and not self._response_api_done:
                try:
                    await conn.response.cancel()
                    logger.debug("Cancelled in-progress response due to barge-in")
                except Exception as e:
                    if "no active response" in str(e).lower():
                        logger.debug("Cancel ignored - response already completed")
                    else:
                        logger.warning("Cancel failed: %s", e)

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("User stopped speaking")
            print("🤔 Processing...")

        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("Assistant response created")
            self._active_response = True
            self._response_api_done = False

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            # event.delta is expected to be PCM16 raw bytes (same as original)
            ap.queue_audio(event.delta)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("Assistant finished speaking")
            print("🎤 Ready for next input...")

        elif event.type == ServerEventType.RESPONSE_DONE:
            logger.info("Response complete")
            self._active_response = False
            self._response_api_done = True

        elif event.type == ServerEventType.ERROR:
            msg = getattr(event, "error", None)
            msg_text = msg.message if msg else str(event)
            if "Cancellation failed: no active response" in str(msg_text):
                logger.debug("Benign cancellation error: %s", msg_text)
            else:
                logger.error("❌ VoiceLive error: %s", msg_text)
                print(f"Error: {msg_text}")

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug("Conversation item created: %s", event.item.id)

        else:
            logger.debug("Unhandled event type: %s", event.type)


# ------------------------------
# Helpers & CLI
# ------------------------------
async def write_conversation_log(message: str) -> None:
    def _write_to_file():
        with open(f"logs/{logfilename}", "a", encoding="utf-8") as conversation_log:
            conversation_log.write(message + "\n")

    await asyncio.to_thread(_write_to_file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Basic Voice Assistant using Azure VoiceLive SDK (sounddevice)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key. If not provided, will use AZURE_VOICELIVE_API_KEY environment variable.",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )

    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_ENDPOINT", "https://your-resource-name.services.ai.azure.com/"),
    )

    parser.add_argument(
        "--agent_id",
        help="Foundry agent ID to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_AGENT_ID", ""),
    )

    parser.add_argument(
        "--foundry_project_name",
        help="Foundry project name to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_PROJECT_NAME", ""),
    )

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant. E.g. alloy, echo, fable, en-US-AvaNeural, en-US-GuyNeural",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-AriaNeural"),
    )

    parser.add_argument(
        "--use-token-credential",
        help="Use Azure token credential instead of API key (default True)",
        action="store_true",
        default=True,
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate credentials
    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Please provide an API key using --api-key or set AZURE_VOICELIVE_API_KEY environment variable,")
        print("or use --use-token-credential for Azure authentication.")
        sys.exit(1)

    if args.use_token_credential:
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    assistant = BasicVoiceAssistant(
        endpoint=args.endpoint,
        credential=credential,
        agent_id=args.agent_id,
        foundry_project_name=args.foundry_project_name,
        voice=args.voice,
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run assistant
    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\n👋 Voice assistant shut down. Goodbye!")
    except Exception as e:
        print("Fatal Error: ", e)


if __name__ == "__main__":
    # Run a quick device diagnostic for user visibility (optional)
    try:
        print("Device list (sounddevice):")
        for i, d in enumerate(sd.query_devices()):
            print(f"{i}: {d['name']} - inputs:{d.get('max_input_channels',0)} outputs:{d.get('max_output_channels',0)} default_sr:{d.get('default_samplerate')}")
        print("Default devices:", sd.default.device)
    except Exception:
        logger.debug("Could not list devices", exc_info=True)

    print("🎙️  Basic Voice Assistant with Azure VoiceLive SDK (sounddevice)")
    print("=" * 50)
    main()
