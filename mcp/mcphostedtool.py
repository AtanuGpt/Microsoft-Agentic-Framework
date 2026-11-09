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

MICROSOFT_LEARN_MCP_URL = "https://learn.microsoft.com/api/mcp"


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
        name="MicrosoftLearn",
        url=MICROSOFT_LEARN_MCP_URL,
        approval_mode="never_require"
        #header_provider=lambda _ctx: {"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
    )

    await mcp_tool.connect()

    print("\nAvailable Microsoft Learn MCP Functions:")
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
            "You are an intelligent Microsoft technology assistant. "
            "Use Microsoft Learn MCP tools to search and retrieve "
            "official Microsoft documentation. "
            "Use microsoft_docs_search for documentation searches. "
            "Use microsoft_docs_fetch for retrieving articles. "
            "Use microsoft_code_sample_search for code examples. "
            "Provide technically accurate responses."
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
