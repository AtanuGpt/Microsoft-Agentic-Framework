import os
import asyncio
from dotenv import load_dotenv
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import ConcurrentBuilder
from agent_framework import AgentExecutorResponse
from azure.identity import AzureCliCredential

load_dotenv(override=True)

instructions_agent01 = "You are an intelligent agent who gets the precise city and country of birth of any cricket player. " \
                       "Just give the name of the city and country only and nothing else."

instructions_agent02 = "You are an intelligent agent who gets the precise climatic conditions of the city of birth of any cricket player. " \
                       "Just give a summary of the climatic conditions and nothing else."

instructions_agent03 = "You are an intelligent agent who gets the best restaurants in the city of birth of any cricket player. " \
                       "Just give the list of top 5 restaurants and nothing else."

instructions_agent04 = "You are a helpful assistant that consolidates multiple domain expert outputs " \
                       "into one cohesive, concise summary with clear takeaways. Keep it under 200 words."

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

summarizer_agent = chat_client.as_agent(
    instructions=instructions_agent04,
    name="summarizer"
)

async def summarize_results(results: list[AgentExecutorResponse]) -> str:
    # Extract one final assistant message per agent
    expert_sections: list[str] = []
    for r in results:
        try:
            messages = getattr(r.agent_response, "messages", [])
            final_text = messages[-1].text if messages and hasattr(messages[-1], "text") else "(no content)"
            expert_sections.append(f"{r.executor_id}:\n{final_text}")
        except Exception as e:
            expert_sections.append(f"{r.executor_id}: (error: {type(e).__name__}: {e})")

    # Ask the model to synthesize a concise summary of the experts' outputs
    prompt = "\n\n".join(expert_sections)
    response = await summarizer_agent.run(prompt)
    # Return the model's final assistant text as the completion result
    return response.messages[-1].text if response.messages else ""

workflow = (
    ConcurrentBuilder(participants=[city_finder, climate_finder, restaurant_finder])
    .with_aggregator(summarize_results)
    .build()
)

async def main():
    events = await workflow.run("Sachin Tendulkar")
    outputs = events.get_outputs()
    print(outputs[0])    


if __name__ == "__main__":
    asyncio.run(main())
