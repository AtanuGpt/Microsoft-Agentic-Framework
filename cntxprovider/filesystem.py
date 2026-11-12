from agent_framework import AgentSession, FileHistoryProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

history = FileHistoryProvider(storage_path="./conversations/")

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

agent = chat_client.as_agent(
    name="Assistant",
    instructions="You are a friendly assistant.",
    context_providers=[history]
)


async def main():    

    session = AgentSession(session_id="atanu") #debjani

    first = await agent.run("My name is Atanu.", session=session)
    print(first.text)

    second = await agent.run("What is my name?", session=session)
    print(second.text)


if __name__ == "__main__":
    asyncio.run(main())
