import asyncio
import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.foundry import FoundryChatClient

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL")
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")

CITY_WEATHER_FORECAST_MCP_URL = "http://127.0.0.1:8080/mcp"

# --------------------------------------------------
# 2. Initialize Foundry Chat Client
# --------------------------------------------------

def get_chat_client():

    return FoundryChatClient(
        project_endpoint=FOUNDRY_PROJECT_ENDPOINT,
        model=FOUNDRY_MODEL,
        credential=DefaultAzureCredential()
    )


# --------------------------------------------------
# 3. Initialize MCP Server Connection
# --------------------------------------------------

def get_mcp_server():

    return MCPStreamableHTTPTool(
        name="CityWeatherForecastMCP",
        url=CITY_WEATHER_FORECAST_MCP_URL,
        approval_mode="never_require"
    )

# --------------------------------------------------
# 4. Create Weather Agent
# --------------------------------------------------

def create_agent(chat_client, mcp_tool):

    return Agent(
        client=chat_client,
        name="CityWeatherForecastAgent",

        instructions="""
        You are an intelligent city weather assistant.

        You have access to the following MCP tools:

        1. get_city_coordinates(cityname)
           Retrieves the latitude and longitude of a city using the Open-Meteo Geocoding API.

        2. get_city_weather(latitude, longitude)
           Retrieves current weather conditions using the Open-Meteo Forecast API.

        Execution instructions:

        - Identify the city name from the user's query.
        - First invoke get_city_coordinates.
        - Extract latitude and longitude from the JSON response.
        - Next invoke get_city_weather using those coordinates.
        - Interpret the weather API response.
        - Present the results in a concise and structured format.

        Important:
        - Always use the MCP tools for current weather information.
        - Never fabricate coordinates or weather information.
        - If the city cannot be found, inform the user.
        - Include temperature, humidity, wind speed,
          rain, showers and precipitation probability
          whenever those values are available.
        """,

        tools=[mcp_tool]
    )


# --------------------------------------------------
# 5. Execute Agent with Streaming
# --------------------------------------------------

async def execute_agent(query, agent):

    print("\nAgent Response:\n")

    result = await agent.run(query)
    print(result.text)

    print("\n")


# --------------------------------------------------
# 6. Application Entry Point
# --------------------------------------------------

async def main():

    query = input("Enter your query: ") #Can you tell me the weather condition for London

    chat_client = get_chat_client()

    # MCP connection lifecycle managed by async with
    async with get_mcp_server() as mcp_tool:
        # Initialize agent
        async with create_agent(chat_client, mcp_tool) as agent:
            await execute_agent(query, agent)


if __name__ == "__main__":
    asyncio.run(main())
