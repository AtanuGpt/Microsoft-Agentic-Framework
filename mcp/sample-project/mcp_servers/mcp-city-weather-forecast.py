import httpx
import json
import socket
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

WEATHER_CODES = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Freezing Light Drizzle",
    57: "Freezing Dense Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Freezing Light Rain",
    67: "Freezing Heavy Rain",
    71: "Slight Snowfall",
    73: "Moderate Snowfall",
    75: "Heavy Snowfall",
    77: "Snow Grains",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Violent Rain Showers",
    85: "Slight Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Slight Hail",
    99: "Thunderstorm with Heavy Hail"
}

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

@mcp.tool()
async def get_city_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
) -> str:
    """
    Get daily weather forecast for a location.

    Args:
        latitude: Latitude of the city.
        longitude: Longitude of the city.
        forecast_days: Number of forecast days (1-16).
    """

    notify_agent(
        f"[MCP] Fetching {forecast_days}-day weather forecast"
    )

    if not 1 <= forecast_days <= 16:
        return json.dumps({
            "status": "error",
            "message": "forecast_days must be between 1 and 16"
        })

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "wind_speed_10m_max,"
            "sunrise,"
            "sunset"
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
        "wind_speed_unit": "kmh"
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:

            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

        daily = data.get("daily", {})
        units = data.get("daily_units", {})

        forecast = []

        for i, date in enumerate(daily.get("time", [])):

            forecast.append({
                "date": date,
                "weather_code": daily["weather_code"][i],
                "weather_condition": WEATHER_CODES.get(
                    daily["weather_code"][i],
                    "Unknown Weather Condition"
                ),
                "temperature_max": {
                    "value": daily["temperature_2m_max"][i],
                    "unit": units.get("temperature_2m_max")
                },
                "temperature_min": {
                    "value": daily["temperature_2m_min"][i],
                    "unit": units.get("temperature_2m_min")
                },
                "precipitation": {
                    "value": daily["precipitation_sum"][i],
                    "unit": units.get("precipitation_sum")
                },
                "precipitation_probability": {
                    "value": daily["precipitation_probability_max"][i],
                    "unit": units.get(
                        "precipitation_probability_max"
                    )
                },
                "wind_speed_max": {
                    "value": daily["wind_speed_10m_max"][i],
                    "unit": units.get("wind_speed_10m_max")
                },
                "sunrise": daily["sunrise"][i],
                "sunset": daily["sunset"][i]
            })

        notify_agent(
            "[MCP] Weather forecast retrieved successfully"
        )

        result = {
            "status": "success",
            "timezone": data.get("timezone"),
            "forecast_days": len(forecast),
            "forecast": forecast
        }

        return json.dumps(result)

    except (httpx.HTTPError, KeyError, IndexError) as e:

        notify_agent(f"[MCP] Forecast API error: {str(e)}")

        return json.dumps({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
