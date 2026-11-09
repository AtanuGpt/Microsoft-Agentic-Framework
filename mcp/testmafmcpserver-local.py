import asyncio
import os
import sys

from contextlib import AsyncExitStack
from dotenv import load_dotenv

from agent_framework import MCPStdioTool
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL")

MCP_SERVER_SCRIPT = r"<<local folder path>>\mcp\mafagentaslocalmcp.py"

# --------------------------------------------------
# 2. Initialize Foundry Chat Client
# --------------------------------------------------

def get_chat_client():

    return FoundryChatClient(
        model=FOUNDRY_MODEL,
        credential=DefaultAzureCredential()
    )


# --------------------------------------------------
# 3. Register MCP Servers
# --------------------------------------------------

async def register_mcp_servers(stack):

    city_info_tool = await stack.enter_async_context(
        MCPStdioTool(
            name="city_info",
            command= sys.executable,
            args=[MCP_SERVER_SCRIPT]
        )
    )

    print("\nAvailable City Tool Functions:")
    for tool in city_info_tool.functions:
        print(f"Tool Name : {tool.name}")
        print(f"Description: {tool.description}")
        print("-" * 50)    
    
    return [city_info_tool]


# --------------------------------------------------
# 4. Create Agent
# --------------------------------------------------

def create_agent(chat_client, tools):

    return chat_client.as_agent(
        name="UtilityAgent",
        instructions=(
            "You are an intelligent weather assistant. "

            "Use get_temperature whenever the user asks "
            "for current temperature or live weather conditions. "

            "Use ask_weather for general weather-related questions. "

            "Always use the appropriate MCP tool instead of "
            "inventing current weather information. "

            "Present the results in a concise and structured format."
        ),
        tools=tools
    )


# --------------------------------------------------
# 5. Execute Agent
# --------------------------------------------------

async def execute_agent(query, agent):

    result = await agent.run(query)
    print(result.text)

# --------------------------------------------------
# 6. Application Entry Point
# --------------------------------------------------

async def main():
    query = input("Enter your query: ") # What is the current time in Kolkata, London, Helsinki and New York? Show in respect to IST. Also, calculate the sum of 1234 and 5678.
    chat_client = get_chat_client()

    async with AsyncExitStack() as stack:
        tools = await register_mcp_servers(stack)
        agent = create_agent(chat_client, tools)

        await execute_agent(query, agent)

if __name__ == "__main__":
    asyncio.run(main())
