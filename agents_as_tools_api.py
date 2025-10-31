from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# Define instructions
instructions_agent01 = (
    "You are an intelligent agent who gets the precise city and country of birth "
    "of any cricket player. Just give the name of the city and country only and nothing else."
)

instructions_agent02 = (
    "You are a specialized agent in giving expert budget-friendly travel itinerary for a given place or country. "
    "Only recommend places, hotels, transport, and restaurant suggestions which will make the plan better."
)

# Create the agents
credential = AzureCliCredential()

agent01 = AzureOpenAIChatClient(credential=credential).create_agent(
    name="CricketerPlaceOfBirth",
    description="An agent that gets the precise city and country of birth of any cricket player.",
    instructions=instructions_agent01,
)

agent02 = AzureOpenAIChatClient(credential=credential).create_agent(
    name="TravelItineraryAdvisor",
    description="An agent that gives expert budget-friendly travel itineraries for a given place or country.",
    instructions=instructions_agent02,
)

main_agent = AzureOpenAIChatClient(credential=credential).create_agent(
    instructions="You are a helpful assistant. Always use the tools provided to you to get the information you need.",
    tools=[agent01.as_tool(), agent02.as_tool()],
)

# Initialize FastAPI app
app = FastAPI(title="Agentic Travel Itinerary API")

@app.get("/")
def home():
    return {"message": "Agentic Travel Itinerary API is running. Use /agent?message=your_query"}

@app.get("/agent")
async def run_agent(message: str = Query(..., description="User input message for the agent")):
    """
    Endpoint that takes a message and streams the agent's response.
    Example: /agent?message=Suggest+me+a+1N/2D+itinerary+for+Rahul+Dravid's+birthplace
    """

    async def stream_response():
        async for update in main_agent.run_stream(message):
            if update.text:
                yield update.text

    return StreamingResponse(stream_response(), media_type="text/plain")
