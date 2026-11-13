
import asyncio

from agent_framework.openai import OpenAIChatClient

MODEL = "qwen2.5-0.5b-instruct-generic-cpu:4"

async def main():

    chat_client = OpenAIChatClient(
        base_url="http://127.0.0.1:56472/v1",
        api_key="none",
        model=MODEL
    )

    agent = chat_client.as_agent(
        name="AzureArchitect",
        instructions="You are an experienced Azure Cloud Architect."
    )

    response = await agent.run(
        "Explain Azure Landing Zones and their key design principles."
    )

    print("\nAssistant:")
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
