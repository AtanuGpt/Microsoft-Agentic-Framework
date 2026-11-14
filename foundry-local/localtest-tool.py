
from openai import OpenAI
import json

MODEL = "<<your local model:version>>"
BASE_URL = "<<foundry local host url>>/v1"


client = OpenAI(
    base_url=BASE_URL,
    api_key="not-needed"
)

# -----------------------------
# Tool implementations
# -----------------------------

def get_weather(location: str):
    # Simulated response - replace with a real weather API
    return {
        "location": location,
        "temperature": "28 C",
        "condition": "Partly Cloudy"
    }


def get_azure_service_info(service: str):
    services = {
        "aks": "Azure Kubernetes Service is a managed Kubernetes platform.",
        "apim": "Azure API Management provides API publishing, security and governance.",
        "cosmosdb": "Azure Cosmos DB is a globally distributed NoSQL database."
    }

    return {
        "service": service,
        "description": services.get(
            service.lower(),
            "Service information not found."
        )
    }


# -----------------------------
# Tool definitions
# -----------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_azure_service_info",
            "description": "Get information about an Azure service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Azure service name"
                    }
                },
                "required": ["service"]
            }
        }
    }
]


# -----------------------------
# Tool registry
# -----------------------------

tool_registry = {
    "get_weather": get_weather,
    "get_azure_service_info": get_azure_service_info
}


# -----------------------------
# Agent execution
# -----------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are an intelligent assistant. "
            "Use the available tools when required. "
            "Do not invent tool results."
        )
    },
    {
        "role": "user",
        "content": "Explain AKS and tell me the weather in Kolkata."
    }
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0
)

assistant_message = response.choices[0].message

# Preserve assistant tool-call message in conversation
messages.append(assistant_message.model_dump(exclude_none=True))

if assistant_message.tool_calls:

    for tool_call in assistant_message.tool_calls:

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"\nCalling tool: {function_name}")
        print(f"Arguments: {arguments}")

        function = tool_registry.get(function_name)

        if function is None:
            result = {"error": "Unknown tool"}
        else:
            result = function(**arguments)

        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Send tool results back to the LLM
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )

    print("\nFinal Assistant Response:")
    print(final_response.choices[0].message.content)

else:
    print(assistant_message.content)
