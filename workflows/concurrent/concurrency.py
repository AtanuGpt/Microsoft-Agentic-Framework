import os
import asyncio
from dotenv import load_dotenv
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import ConcurrentBuilder
from agent_framework import AgentResponse
from azure.identity import AzureCliCredential

load_dotenv(override=True)

instructions_agent01 = "You are an intelligent agent who gets the precise city and country of birth of any cricket player. " \
                       "Just give the name of the city and country only and nothing else."

instructions_agent02 = "You are an intelligent agent who gets the precise climatic conditions of the city of birth of any cricket player. " \
                       "Just give a summary of the climatic conditions and nothing else."

instructions_agent03 = "You are an intelligent agent who gets the best restaurants in the city of birth of any cricket player. " \
                       "Just give the list of top 5 restaurants and nothing else."

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

city_finder = chat_client.as_agent(
    instructions=instructions_agent01,
    name="city_finder"
)

climate_finder = chat_client.as_agent(
    instructions=instructions_agent02,
    name="climate_finder"
)

restaurant_finder = chat_client.as_agent(
    instructions=instructions_agent03,
    name="restaurant_finder"
)

async def main():
    workflow = ConcurrentBuilder(
        participants=[city_finder, climate_finder, restaurant_finder]
    ).build()

    events = await workflow.run("Don Bradman")
    outputs = events.get_outputs()

    if outputs:
        print("===== Final Aggregated Results =====")
        final: AgentResponse = outputs[0]
        for msg in final.messages:
            name = msg.author_name or "assistant"
            print(f"{'-' * 60}\n\n[{name}]:\n{msg.text}")


if __name__ == "__main__":
    asyncio.run(main())
