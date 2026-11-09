import asyncio
import os

from dotenv import load_dotenv

from agent_framework import MCPStreamableHTTPTool
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

FOUNDRY_MODEL = os.getenv("FOUNDRY_MODEL")

MICROSOFT_LEARN_MCP_URL = "http://127.0.0.1:8080/mcp"


# --------------------------------------------------
# 2. Initialize Foundry Chat Client
# --------------------------------------------------

def get_chat_client():

    return FoundryChatClient(
        model=FOUNDRY_MODEL,
        credential=DefaultAzureCredential()
    )


# --------------------------------------------------
# 3. Initialize Hosted MCP Server
# --------------------------------------------------

async def get_mcp_server():

    mcp_tool = MCPStreamableHTTPTool(
        name="CityInfoMCPTool",
        url=MICROSOFT_LEARN_MCP_URL,
        approval_mode="never_require"
    )

    await mcp_tool.connect()

    print("\nAvailable City Info Functions:")
    print("=" * 60)

    for tool in mcp_tool.functions:
        print(f"Tool Name  : {tool.name}")
        print(f"Description: {tool.description}")
        print("-" * 60)

    return mcp_tool


# --------------------------------------------------
# 4. Create Agent
# --------------------------------------------------

def create_agent(chat_client, mcp_tool):

    return chat_client.as_agent(
        name="MicrosoftLearnAgent",
        instructions=(
            "You are an intelligent weather assistant. "
           
            "Use get_temperature whenever the user asks "
            "for current temperature or live weather conditions. "

            "Use ask_weather for general weather-related questions. "

            "Always use the appropriate MCP tool instead of "
            "inventing current weather information. "

            "Present the results in a concise and structured format."
        ),
        tools=[mcp_tool]
    )


# --------------------------------------------------
# 5. Execute Agent with Streaming
# --------------------------------------------------

async def execute_agent(query, agent):
    print("\nAgent Response:\n")

    async for chunk in agent.run(query, stream=True):

        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n")


# --------------------------------------------------
# 6. Application Entry Point
# --------------------------------------------------

async def main():
    query = input("Enter your query: ") #Explain how Azure AI Search integrates with Azure OpenAI for implementing RAG applications.
    chat_client = get_chat_client()
    mcp_tool = await get_mcp_server()

    async with mcp_tool:

        agent = create_agent(chat_client, mcp_tool)

        await execute_agent(query, agent)


if __name__ == "__main__":
    asyncio.run(main())
