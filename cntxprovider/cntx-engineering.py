import asyncio
import os
import httpx

from dotenv import load_dotenv
from azure.identity import AzureCliCredential

from agent_framework import Agent, ContextProvider, Message
from agent_framework.foundry import FoundryChatClient

load_dotenv()

# --------------------------------------------------
# 1. Foundry Chat Client
# --------------------------------------------------

client = FoundryChatClient(
    credential=AzureCliCredential(),
    project_endpoint=os.getenv("FOUNDRY_PROJECT_ENDPOINT"),
    model=os.getenv("FOUNDRY_MODEL")
)

# -----------------------------------------------------------
# 2. Using LLM to extract the  city name from the user query
# -----------------------------------------------------------

async def extract_city(user_query):

    result = await client.get_response(
        messages=[
            Message(
                role="system",
                contents=[
                    """
                    Extract the city name from the user's query.
                    Return only the city name.
                    If no city is identified, return NONE.
                    """
                ]
            ),
            Message(
                role="user",
                contents=[user_query]
            )
        ]
    )

    return result.text.strip()

# --------------------------------------------------
# 3. Context Provider
# --------------------------------------------------

class GeoLocationContextProvider(ContextProvider):

    def __init__(self):
        super().__init__("geolocation_context")

    async def before_run(self, *, context, **kwargs):

        print("GeoLocationContextProvider invoked!")     

        # 1. Read the user's query
        user_query = context.input_messages[-1].text

        print(f"User Query: {user_query}")

        # Extract city using LLM
        cityname = await extract_city(user_query)     

        print(f"Extracted city: {cityname}")

        # 3. Retrieve geographical information
        url = "https://geocoding-api.open-meteo.com/v1/search"

        params = {
            "name": cityname,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])

        if not results:
            print("Location not found")
            return

        location = results[0]

        # 4. Build geographical context
        result_context = f"""
            City: {location['name']}
            Country: {location['country']}
            Latitude: {location['latitude']}
            Longitude: {location['longitude']}
            Timezone: {location.get('timezone')}
        """

        # 5. Inject context into MAF
        context.extend_messages(
            self.source_id,
            [
                Message(
                    role="system",
                    contents=[result_context]
                )
            ]
        )

        print("Geographical context injected successfully")

# --------------------------------------------------
# 4. Create Agent with Context Provider
# --------------------------------------------------

agent = Agent(
    client=client,
    name="WeatherContextAgent",
    instructions="""
        You are a geographical and weather information assistant.
        Use the externally supplied geographical context when answering location-related questions.
        If coordinates are unavailable, ask the user to clarify the location.
        Do not fabricate geographical information.
    """,
    context_providers=[
        GeoLocationContextProvider()
    ]
)

# --------------------------------------------------
# 5. Execute
# --------------------------------------------------

async def main():
    result = await agent.run("What is the georgraphical location of Mumbai")
    print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
