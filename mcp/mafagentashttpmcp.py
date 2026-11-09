import os

from dotenv import load_dotenv

from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential
from mcp.server.mcpserver import MCPServer

load_dotenv()

FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL")

chat_client = FoundryChatClient(
    model=FOUNDRY_MODEL, #gpt-4o-mini
    credential=DefaultAzureCredential()
)

city_agent = chat_client.as_agent(
    name="CityInfoAgent",
    instructions="You are an intelligent agent who gets the precise details of any city in the world. "
)

mcp = MCPServer("city-info-server")

@mcp.tool()
async def get_weather(question: str) -> str:
    """Return the climatic condition for a city."""
    result = await city_agent.run(question)
    return result.text

@mcp.tool()
async def get_hotels(question: str) -> str:
    """Return top hotels in a city."""
    result = await city_agent.run(question)
    return result.text

@mcp.tool()
async def get_restaurants(question: str) -> str:
    """Return top restaurants in a city."""
    result = await city_agent.run(question)
    return result.text


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
