from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

async def coingecko_crypto_mcp_server(question: str) -> str:
    """HTTP-based CoinGecko MCP server."""
    async with (
        AzureCliCredential() as credential,
        MCPStreamableHTTPTool(
            name="CoinGecko MCP",
            url="https://mcp.api.coingecko.com/mcp"
        ) as mcp_server,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            name="CoinGeckoAgent",
            instructions="You help with crypo currency questions. Convert any amount to rupees (INR) only.",
            store= True
        ) as agent,
    ):
        result = await agent.run(question, tools=mcp_server)        
        return result.text
