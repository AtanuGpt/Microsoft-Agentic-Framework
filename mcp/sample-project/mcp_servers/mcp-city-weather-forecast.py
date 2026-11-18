import httpx
import json
import socket
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

def notify_agent(message: str):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(
            message.encode("utf-8"),
            ("127.0.0.1", 8765)
        )

mcp = MCPServer("city-weather-forecast-server")

@mcp.tool()
async def get_city_coordinates(cityname: str) -> str:
    """Return the latitude and longitude of a city in JSON format."""

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": cityname,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    notify_agent(f"[MCP] Looking up coordinates for {cityname}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        if not data.get("results"):
            return json.dumps({
                "city": cityname,
                "error": "City not found"
            })

        city = data["results"][0]

        result = {
            "status": "success",
            "message": f"Coordinates retrieved for {cityname}",
            "latitude": city["latitude"],
            "longitude": city["longitude"]
        }
        
        notify_agent(f"[MCP] Coordinates retrieved: {result}")

        return json.dumps(result)

    except httpx.HTTPError as e:
        return json.dumps({
            "city": cityname,
            "error": str(e)
        })

@mcp.tool()
async def get_city_weather(latitude: float, longitude: float) -> str:
    """Get current weather details using latitude and longitude."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "wind_speed_10m,"
            "rain,"
            "showers,"
            "relative_humidity_2m,"
            "precipitation_probability"
        ),
        "timezone": "auto"
    }

    notify_agent("[MCP] Fetching current weather information")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        current = data.get("current", {})
        units = data.get("current_units", {})

        weather = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "time": current.get("time"),
            "temperature": {
                "value": current.get("temperature_2m"),
                "unit": units.get("temperature_2m")
            },
            "humidity": {
                "value": current.get("relative_humidity_2m"),
                "unit": units.get("relative_humidity_2m")
            },
            "wind_speed": {
                "value": current.get("wind_speed_10m"),
                "unit": units.get("wind_speed_10m")
            },
            "rain": {
                "value": current.get("rain"),
                "unit": units.get("rain")
            },
            "showers": {
                "value": current.get("showers"),
                "unit": units.get("showers")
            },
            "precipitation_probability": {
                "value": current.get("precipitation_probability"),
                "unit": units.get("precipitation_probability")
            }
        }

        notify_agent("[MCP] Weather information retrieved successfully")

        return json.dumps(weather)

    except httpx.HTTPError as e:
        return json.dumps({
            "error": str(e)
        })

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
