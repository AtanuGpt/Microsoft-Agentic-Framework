from agent_framework import AgentSession
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

agent = chat_client.as_agent(
    name="Assistant",
    instructions="You are a friendly assistant.",
)

async def main():    

    session = AgentSession()

    stream = agent.run("Tell me a short summary about Gen AI.", session=session, stream=True)
    print("Assistant: ", end="", flush=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)
    print()

    await stream.get_final_response()


if __name__ == "__main__":
    asyncio.run(main())
