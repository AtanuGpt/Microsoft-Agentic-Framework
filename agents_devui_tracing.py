import asyncio
from operator import truediv
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from agent_framework.devui import serve

instructions_agent01 = "You are an intelligent agent who gets the precise city and country of birth of any cricket player. " \
                       "Just give the name of the city and country only and nothing else."

instructions_agent02 = "You are a specialized agent in giving expert budget friendly travel itinerary for a given place or country. " \
                       "Only recommend places, hotels, transport and restaurants suggesations which will in making the plan better; "

agent01 = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    name="CricketerPlaceOfBirth",
    description="An agent that gets the precise city and country of birth of any cricket player.",
    instructions=instructions_agent01,
 )

agent02 = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent( 
    name="TravelItineraryAdvisor", 
    description="An agent that gets can give expert budget friendly travel itinerary for a given place or country",
    instructions = instructions_agent02
 )

main_agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    name="HelpfulTravelAssistant",
    instructions="You are a helpful assistant. Always use the tools provided to you to get the information you need.",
    tools=[agent01.as_tool(), agent02.as_tool()]
)

def main():
    #message = "Suggest me a 1N/2D travel itinerary for that city of birth of cricketer Sourav Ganguly"  -- Not required anymore as you can pass it directly in the devui window  
    serve(entities=[main_agent], auto_open=True, tracing_enabled=True, port=8090)


# Run the async function using asyncio
if __name__ == "__main__":
    main()

