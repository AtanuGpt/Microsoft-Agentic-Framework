import asyncio
import os

from agent_framework.openai import OpenAIChatClient
from agent_framework.observability import configure_otel_providers

MODEL = "<<your local model:version>>"
BASE_URL = "<<local endpoint>>/v1"

# The telemetry will show up in the Foundry Toolkit >> Developer Tools >> Monitor >> Tracing .. All under VS Code

async def main():

    # Send telemetry to VS Code Foundry Toolkit
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
    os.environ["OTEL_SERVICE_NAME"] = "FoundryLocal-AzureArchitect"

    configure_otel_providers(
        enable_sensitive_data=True #To see the query and the answer provided else False to see only the metadata
    )

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

    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
