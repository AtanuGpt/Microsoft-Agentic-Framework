from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

async def ms_learn_mcp_server(question : str) -> str:
    """HTTP-based Microsoft Azure MCP server."""
    async with (
        AzureCliCredential() as credential,
        MCPStreamableHTTPTool(
            name="MS Learn MCP",
            url="https://learn.microsoft.com/api/mcp"
        ) as mcp_server,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            name="MSLearnAgent",
            instructions="You help with Microsoft Azure documentation questions.",
            store=True
        ) as agent,
    ):
        result = await agent.run(question, tools=mcp_server)        
        return result.text
