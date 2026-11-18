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

#For receiving socket logs
class MCPLogReceiver(asyncio.DatagramProtocol):
    def datagram_received(self, data, addr):
        message = data.decode("utf-8", errors="replace")
        print(message, flush=True)


async def start_mcp_log_receiver():
    loop = asyncio.get_running_loop()

    transport, protocol = await loop.create_datagram_endpoint(
        MCPLogReceiver,
        local_addr=("127.0.0.1", 8765)
    )

    print("[Agent] MCP log receiver initialized.")
    return transport

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
        You are an intelligent city weather/climate assistant.

        You MUST invoke the MCP tools. DO NOT answer any queries directly from your internal knowledge.

        You have access to the following MCP tools:

        1. get_city_coordinates(cityname)
           Retrieves the latitude and longitude of a city using the Open-Meteo Geocoding API.

        2. get_city_weather(latitude, longitude)
           Retrieves current weather conditions using the Open-Meteo Forecast API.

        3. get_city_forecast(latitude, longitude, forecast_days)
           Retrieves daily weather forecasts for a specified number of days, up to 14 days.

           When presenting multi-day weather forecasts, ALWAYS use a Markdown table instead of individual date headings.

            Use the following table structure:

            | Date | Min Temp | Max Temp | Rain     | Rain Probability | Wind Speed |
            |------|----------|----------|----------|------------------|------------|
            | Date | Value °C | Value °C | Value mm | Value %          | Value km/h |

        Execution instructions:

        - Identify the city name from the user's query.
        - First invoke get_city_coordinates.
        - Extract latitude and longitude from the JSON response.
        - Next invoke get_city_weather using those coordinates.
        - Interpret the weather API response.
        - Present the results in a concise and structured format.

        Mandatory:
        - Always use the MCP tools for current weather information.
        - Never fabricate coordinates or weather information.
        - If the city cannot be found, inform the user
        - Include temperature, humidity, wind speed, rain, showers and precipitation probability whenever those values are available.
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
     # Start MCP log receiver
    log_transport = await start_mcp_log_receiver()

    try:

        query = input("Enter your query: ")

        chat_client = get_chat_client()

        async with get_mcp_server() as mcp_tool:

            agent = create_agent(chat_client, mcp_tool)
            result = await agent.run(query)

            print("\nAgent Response:\n")
            print(result.text)

    finally:
        log_transport.close()


if __name__ == "__main__":
    asyncio.run(main())
