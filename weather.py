import requests

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# note: used Github Copilot to search and generate the following weather code mapping from OpenMeteo

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm"
}

def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "fahrenheit",
        "current_weather": True,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=10)
    data = r.json()
    weather = data.get("current_weather")
    if weather:
        temp = weather.get("temperature")
        code = int(weather.get("weathercode", 0))
        condition = WEATHER_CODES.get(code)
        return temp, condition
    return None

if __name__ == "__main__":
    # using a hardcoded location for LA
    latitude = 34.0549
    longitude = -118.2426
    result = get_weather(latitude, longitude)
    if result:
        temp, condition = result
        print(f"Location: Los Angeles")
        print(f"Temperature: {temp} °F")
        print(f"Weather: {condition}")

