import os
import asyncio

from dotenv import load_dotenv

from agent_framework import tool
from agent_framework import AgentSession
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

load_dotenv()

FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL")

chat_client = FoundryChatClient(
    model=FOUNDRY_MODEL, #gpt-4o-mini
    credential=DefaultAzureCredential()
)

@tool(approval_mode="never_require")
async def get_weather(location: str) -> str:
    """Return the climatic condition for a city."""
    result = f"The weather in {location} is sunny."
    return result


@tool(approval_mode="never_require")
async def get_hotels(location: str) -> str:
    """Return top hotels in a city."""
    result = (
        f"Top 3 hotels in {location} are "
        "1. Hotel ONE, 2. Hotel TWO, 3. Hotel THREE."
    )
    return result


@tool(approval_mode="never_require")
async def get_restaurants(location: str) -> str:
    """Return top restaurants in a city."""
    result = (
        f"Top 3 restaurants in {location} are "
        "1. Restaurant ONE, 2. Restaurant TWO, 3. Restaurant THREE."
    )
    return result


city_agent = chat_client.as_agent(
    name="CityInfoAgent",
    instructions="You are an intelligent agent who gets the precise details of any city in the world",
    tools=[get_weather, get_hotels, get_restaurants]
)

async def main():  
    session = AgentSession()

    stream = city_agent.run("Tell me about the city of Mumbai, India. I want to know the weather, top hotels, and top restaurants.", session=session, stream=True)
    print("Assistant: ", end="", flush=True)

    seen_tool_calls = set()
    streamed_text = ""

    async for update in stream:

        for content in update.contents:
            if content.type == "function_call":
                name = content.name
                arguments = content.arguments

                call_key = (name, str(arguments))

                #Remvoing duplicate tool calls from being printed
                if call_key not in seen_tool_calls:
                    seen_tool_calls.add(call_key)
                    print(
                        f"\n[Calling {name}({arguments})]",
                        flush=True

                    )

        if update.text:
            chunk = update.text

            # Avoid printing duplicate text chunks that may occur due to streaming updates
            if streamed_text and chunk.startswith(streamed_text):
                continue

            streamed_text += chunk
            print(chunk, end="", flush=True)

    # if you skip the call, session may not be fully updated and the stream object holds resources until it is finalized
    await stream.get_final_response()

if __name__ == "__main__":
    asyncio.run(main())
