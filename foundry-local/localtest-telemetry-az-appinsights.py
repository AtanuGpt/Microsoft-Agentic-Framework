import asyncio
import os

from dotenv import load_dotenv

from agent_framework.openai import OpenAIChatClient
from microsoft.opentelemetry import use_microsoft_opentelemetry

load_dotenv()

MODEL = "your local  model:version"
BASE_URL = "local url/v1"


async def main():

    # Application Insights configuration
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not connection_string:
        raise ValueError(
            "Application Insights connection string not configured"
        )

    # Configure OpenTelemetry with Azure Monitor exporter
    use_microsoft_opentelemetry(
        enable_azure_monitor=True,
        azure_monitor_connection_string=connection_string,
        enable_sensitive_data=True
    )

    # Initialize local model
    chat_client = OpenAIChatClient(
        base_url=BASE_URL,
        api_key="none",
        model=MODEL
    )

    # Create agent
    agent = chat_client.as_agent(
        name="AzureArchitect",
        instructions="You are an experienced Azure Cloud Architect."
    )

    # Execute agent
    response = await agent.run(
        "Explain Gen AI in 3 bullet points."
    )

    print("\nAssistant:")
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
