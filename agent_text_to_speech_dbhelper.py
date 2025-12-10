import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import numpy as np
import wave
import io
import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity import AzureCliCredential

# ---------------------------------------------------------------
# TEXT → SPEECH (Cognitive Services) .. read from environment file
# -----------------------------------------------------------------
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
VOICE_NAME = "en-US-AriaNeural"   # change voice if needed

def play_wav_bytes(wav_bytes: bytes):
    """Play wav audio returned by Azure Speech."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        audio = np.frombuffer(pcm, dtype=np.int16)

        if wf.getnchannels() > 1:
            audio = audio.reshape((-1, wf.getnchannels()))

        sd.play(audio, wf.getframerate())
        sd.wait()

def speak_text(text: str):
    """Use Azure Cognitive Speech to speak a text."""
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = VOICE_NAME

    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass
    else:
        print("❌ Speech synthesis failed:", result.reason)


# ---------------------------------------------
# QUERY AGENT (returns text only once)
# ---------------------------------------------
async def query_agent(question: str) -> str:
    async with ChatAgent(
        chat_client=AzureAIAgentClient(
            async_credential=AzureCliCredential(),
            agent_id="<<copy and paste your agent id from Microsoft Foundry agents"
        ),
        instructions="You are a helpful assistant."
    ) as agent:

        result = await agent.run(question)

        # RETURN text only — no printing inside this function
        return result.text


# ---------------------------------------------
# CHAT LOOP (runs continuously)
# ---------------------------------------------
async def chat_loop():

    print("\n=============================")
    print("🔊 Voice-enabled Azure Agent")
    print("Type your questions. Ctrl+C to exit.")
    print("=============================\n")

    while True:
        # get user question
        question = await asyncio.to_thread(input, "You: ")

        if not question.strip():
            continue

        print("\n🤖 Thinking...\n")

        # ask agent
        reply = await query_agent(question)

        # print once only
        print(f"Agent: {reply}\n")

        # speak once only
        speak_text(reply)


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
