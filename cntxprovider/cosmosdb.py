from agent_framework import AgentSession, FileHistoryProvider
from agent_framework.azure import CosmosHistoryProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

cosmos_history = CosmosHistoryProvider(
    endpoint=os.environ["COSMOS_ENDPOINT"],
    credential=os.environ["COSMOS_KEY"],
    database_name=os.environ["COSMOS_DATABASE_NAME"],
    container_name=os.environ["COSMOS_CONTAINER_NAME"]
)

print("Cosmos History Provider initialized successfully")

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

agent = chat_client.as_agent(
    name="Assistant",
    instructions="You are a friendly assistant.",
    context_providers=[cosmos_history],
    default_options={"store": False}
)

async def main():    

    async with cosmos_history, agent:

        session = AgentSession(session_id="atanu") #debjani

        first = await agent.run("My new name is Atanu Gupta", session=session)
        print(first.text)

        second = await agent.run("What is my name?", session=session)
        print(second.text)


if __name__ == "__main__":
    asyncio.run(main())
