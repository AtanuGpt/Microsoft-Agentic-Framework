import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

async def ms_learn_mcp_server():
    """HTTP-based CoinGecko MCP server."""
    async with (
        AzureCliCredential() as credential,
        MCPStreamableHTTPTool(
            name="MS Learn MCP",
            url="https://learn.microsoft.com/api/mcp"
        ) as mcp_server,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            name="MSLearnAgent",
            instructions="You help with Microsoft documentation questions.",
            store=True
        ) as agent,
    ):
        async for chunk in agent.run_stream("How to create an Azure kubernetes cluster (AKS)?", tools=mcp_server):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()

asyncio.run(ms_learn_mcp_server())
