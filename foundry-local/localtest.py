
import asyncio

from agent_framework.openai import OpenAIChatClient

MODEL = "<<your local model:version>>"
BASE_URL = "<<foundry local host url>>/v1"

async def main():

    chat_client = OpenAIChatClient(
        base_url=BASE_URL,
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
