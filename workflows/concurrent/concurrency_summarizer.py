import os
import asyncio
from dotenv import load_dotenv
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import ConcurrentBuilder
from agent_framework import AgentResponse, AgentExecutorResponse
from azure.identity import AzureCliCredential

load_dotenv(override=True)

instructions_climate = "You are an intelligent agent who gets the precise climatic conditions of any city " \
                       "Just give a summary of the climatic conditions and nothing else."

instructions_hotels = "You are an intelligent agent who gets the top hotels in any city. " \
                       "Just give the list of top 5 hotels and nothing else."

instructions_restaurants = "You are an intelligent agent who gets the top restaurants in any city. " \
                           "Just give the list of top 5 restaurants and nothing else."

instructions_agent04 = "You are a helpful assistant that consolidates multiple domain expert outputs " \
                       "into one cohesive, concise summary with clear takeaways. Keep it under 200 words."

chat_client = FoundryChatClient(
    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
    model=os.environ["FOUNDRY_MODEL"], #gpt-4o-mini
    credential=AzureCliCredential()
)

climate_finder = chat_client.as_agent(
    instructions=instructions_climate,
    name="climate_finder"
)

hotel_finder = chat_client.as_agent(
    instructions=instructions_hotels,
    name="hotel_finder"
)

restaurant_finder = chat_client.as_agent(
    instructions=instructions_restaurants,
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
    ConcurrentBuilder(participants=[climate_finder, hotel_finder, restaurant_finder])
    .with_aggregator(summarize_results)
    .build()
)

async def main():
    events = await workflow.run("Kolkata, India")
    outputs = events.get_outputs()

    if outputs:
        print("===== Final Aggregated Results =====")
        final: AgentResponse = outputs[0]
        print(final)


if __name__ == "__main__":
    asyncio.run(main())
