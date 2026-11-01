import os
import asyncio
from dotenv import load_dotenv
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from agent_framework import AgentResponse
from azure.identity import AzureCliCredential

load_dotenv(override=True)

instructions_agent01 = "You are an intelligent agent who gets the precise city and country of birth of any cricket player. " \
                       "Just give the name of the city and country only and nothing else." \
                       "Output format: {'city': 'city_name', 'country': 'country_name'}"

instructions_agent02 = "You are a specialized agent in giving expert 1N/2D budget friendly travel itinerary for a given place in a country. " \
                       "Only recommend places, hotels, transport and restaurants suggesations which will in making the plan better and budget friendly. " \
                       "You will get the city and country from the previous agent. Just give the travel itinerary based on that information. " 

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

city_finder = chat_client.as_agent(
    instructions=instructions_agent01,
    name="city_finder"
)

travel_planner = chat_client.as_agent(
    instructions=instructions_agent02,
    name="travel_planner"
)

async def main():
    workflow = SequentialBuilder(
        participants=[city_finder, travel_planner],
        chain_only_agent_responses=True
    ).build()

    events = await workflow.run("Sachin Tendulkar")
    outputs = events.get_outputs()

    if outputs:
        print("===== Final Response =====")
        final: AgentResponse = outputs[0]
        for msg in final.messages:
            name = msg.author_name or "assistant"
            print(f"[{name}]\n{msg.text}")


if __name__ == "__main__":
    asyncio.run(main())
