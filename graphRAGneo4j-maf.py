import asyncio
import os
import json

from dotenv import load_dotenv
from neo4j import GraphDatabase
from azure.identity import AzureCliCredential

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient

# -----------------------------------------
# 1. Load configuration
# -----------------------------------------

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# -----------------------------------------
# 2. Initialize Neo4j driver
# -----------------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# Validate connectivity
driver.verify_connectivity()

print("Connected to Neo4j successfully!")

# -----------------------------------------
# 3. Define Graph Retrieval Tool
# -----------------------------------------

@tool(approval_mode="never_require")
def get_graph_context(question: str) -> str:
    """
        Retrieve relevant information from the enterprise
        knowledge graph to answer questions about entities
        and their relationships.

        Supports:
        - Applications, platforms and databases
        - Application and infrastructure dependencies
        - Direct and indirect relationships
        - Multi-hop relationship traversal
        - Upstream and downstream dependencies
        - Impact and dependency analysis
        - Common dependencies between entities
        - Connected components and relationship paths

        Use this tool whenever a question requires information
        about entities or relationships represented in the
        enterprise knowledge graph.

        The question should be passed as the input.
    """

    cypher_query = """
    MATCH (n)-[r]->(m)
    RETURN
        n.name AS source,
        labels(n) AS source_type,
        type(r) AS relationship,
        m.name AS target,
        labels(m) AS target_type
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        records = session.run(cypher_query)

        results = [dict(record) for record in records]

    return json.dumps(results, indent=2)

# -----------------------------------------
# 4. Initialize MAF Agent
# -----------------------------------------

async def main():

    credential = AzureCliCredential()

    chat_client = FoundryChatClient(
        project_endpoint=os.getenv("FOUNDRY_PROJECT_ENDPOINT"),
        model=os.getenv("FOUNDRY_MODEL"),
        credential=credential
    )

    agent = Agent(
        client=chat_client,
        name="EnterpriseGraphAgent",
        instructions = """
            You are an Enterprise Knowledge Graph Assistant.

            Your responsibility is to answer questions about enterprise entities and their relationships using
            the Neo4j knowledge graph.

            CAPABILITIES:

            1. Identify applications, platforms and databases.
            2. Understand direct and indirect dependencies.
            3. Perform multi-hop relationship reasoning.
            4. Identify upstream and downstream dependencies.
            5. Perform impact and dependency analysis.
            6. Identify common relationships between entities.
            7. Explain dependency chains and relationship paths.

            TOOL USAGE:

            - Use get_graph_context whenever the user asks a question that requires knowledge graph information.
            - Pass the complete user question to the tool.
            - Do not restrict graph retrieval to database-related questions.
            - Do not ask for additional entity names when the graph contains sufficient information to answer.
            - Use the retrieved graph relationships as the source of truth.

            RESPONSE RULES:

            - Never invent entities or relationships.
            - Clearly distinguish direct and indirect dependencies.
            - Explain multi-hop relationships.
            - If information is unavailable in the graph, explicitly mention that limitation.
            - Do not assume relationships that are not represented.
        """,

        tools=[get_graph_context]
    )

    # -------------------------------------
    # 5. Execute user query
    # -------------------------------------

    question = (
        "Which Azure services are exposed through Private Endpoints?"
    )

    print("\nUser:", question)

    response = await agent.run(question)

    print("\nAgent Response:")
    print(response.text)


# -----------------------------------------
# 6. Application entry point
# -----------------------------------------

if __name__ == "__main__":

    try:
        asyncio.run(main())

    finally:
        driver.close()
