
import asyncio
import os

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

    time_tool = await stack.enter_async_context(
        MCPStdioTool(
            name="time",
            command="uvx",
            args=["mcp-server-time"],
        )
    )

    calculator_tool = await stack.enter_async_context(
        MCPStdioTool(
            name="calculator",
            command="uvx",
            args=["mcp-server-calculator"],
        )
    )

    return [time_tool, calculator_tool]


# --------------------------------------------------
# 4. Create Agent
# --------------------------------------------------

def create_agent(chat_client, tools):

    return chat_client.as_agent(
        name="UtilityAgent",
        instructions=(
            "You are an intelligent utility assistant. "
            "Use the time tool for current time and timezone-related queries. "
            "Use the calculator tool for mathematical calculations. "
            "Always use the appropriate tool when required."
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
    query = input("Enter your query: ") #What is the current time in Kolkata, London, Helsinki and New York? Also calculate 55 * 45 + 750.
    chat_client = get_chat_client()

    async with AsyncExitStack() as stack:
        tools = await register_mcp_servers(stack)
        agent = create_agent(chat_client, tools)

        await execute_agent(query, agent)

if __name__ == "__main__":
    asyncio.run(main())
