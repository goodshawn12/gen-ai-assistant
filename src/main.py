import os
import logging
import requests
from datetime import datetime, timezone, timedelta

weather_api_key = os.getenv('REACT_APP_WEATHER_API_KEY')
if weather_api_key is None:
    logging.error("REACT_APP_WEATHER_API_KEY environment variable is not set")

logging.basicConfig(filename='main.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def to_epoch_timestamp(date_time: any) -> int:
    """
    Input Format: YYYY-MM-DDTHH:MM:SS+TZ, e.g., 2024-11-21T10:30:00-08
    If no TZ information available, the code will return the local time zone
    """
    try:
        # Check if input is a string and validate the format
        if isinstance(date_time, str):
            try:
                # Attempt to parse ISO 8601 string with timezone
                date = datetime.fromisoformat(date_time)
            except ValueError:
                # Handle cases where the timezone is missing
                if "T" in date_time and ":" in date_time:
                    logging.warning("Warning: Time Zone info not available, using local time zone")
                    date = datetime.fromisoformat(date_time + "+00:00")
                else:
                    logging.error("Invalid date format. Expected format: 'YYYY-MM-DDTHH:MM:SS+TZ'")
        else:
            # Assume it's a datetime object
            date = date_time

        # Convert to epoch timestamp
        epoch_timestamp = int(date.timestamp())
        return epoch_timestamp
    except Exception as e:
        logging.error(f"Error processing date_time: {e}")


def assess_bp(data_sbp: int, data_dbp: int) -> dict:
    if data_sbp < 90 or data_dbp < 50:
        is_normal_bp = 0
        bp_message = (
            "Thanks for sharing your reading! Your blood pressure reading is abnormal today. "
            "If you haven't done so, could you recheck your blood pressure to ensure the reading is accurate?\n"
        )
    elif data_sbp >= 130 or data_dbp >= 90:
        is_normal_bp = 0
        bp_message = (
            "Thanks for sharing your reading! Your blood pressure is higher than normal today. "
            "Watch for the following symptoms such as dizziness, headache, and chest discomfort. "
            "Contact your provider if needed. Otherwise, recheck your blood pressure after a few minutes of rest.\n"
        )
    else:
        is_normal_bp = 1
        bp_message = "Thanks for sharing your reading! Your blood pressure looks great.\n"

    return {"isNormalBP": is_normal_bp, "bpMessage": bp_message}


def assess_outdoor_env(
    temperature: float, weather_main: str, current_time: datetime, sunrise: datetime, sunset: datetime
) -> dict:
    if sunrise <= current_time <= sunset:
        day_or_night = "daytime"
    else:
        day_or_night = "nighttime"

    is_good_temp = 45 <= temperature <= 90
    is_rain = "rain" in weather_main.lower()
    in_or_out = "indoor" if is_rain or not is_good_temp or day_or_night == "nighttime" else "outdoor"

    return {
        "inOrOut": in_or_out,
        "dayOrNight": day_or_night,
        "isGoodTemp": is_good_temp,
        "isRain": is_rain
    }


def get_weather(date_time: any, city_name: any) -> dict:
    limit = 1
    geo_api_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit={limit}&appid={weather_api_key}"
    lat, lon = None, None

    # Fetch Geo coordinates
    try:
        response = requests.get(geo_api_url)
        response.raise_for_status()
        data = response.json()
        lat = data[0]["lat"]
        lon = data[0]["lon"]
    except (requests.RequestException, IndexError, KeyError) as error:
        logging.error("Error fetching Geo coordinates: %s", error)

    # Default to San Diego coordinates if not found
    if lat is None or lon is None:
        logging.error("Error: Unable to fetch weather data due to missing coordinates. Using San Diego's coordinates as default.")
        lat, lon = 32.7157, -117.1611

    weather_message = "No weather message available"
    weather_data = "No weather data available"
    weather_time = ""

    # Fetch weather data
    try:
        time_stamp = to_epoch_timestamp(date_time)
        api_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={time_stamp}&appid={weather_api_key}&units=imperial"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        data_current = data["data"][0]

        # Extract relevant data
        temperature = data_current["temp"]
        time_zone_offset = data["timezone_offset"]
        current_time = datetime.fromtimestamp(data_current["dt"] + time_zone_offset, tz=timezone.utc)
        sunrise = datetime.fromtimestamp(data_current["sunrise"] + time_zone_offset, tz=timezone.utc)
        sunset = datetime.fromtimestamp(data_current["sunset"] + time_zone_offset, tz=timezone.utc)
        time_zone = data["timezone"]
        uvi = data_current["uvi"]
        weather_main = data_current["weather"][0]["main"]

        # Assess outdoor environment
        in_or_out, day_or_night, _, _ = assess_outdoor_env(temperature, weather_main, current_time, sunrise, sunset).values()

        # Construct weather message and data
        weather_message = f"{in_or_out}, {day_or_night}"
        weather_time = f"{current_time.isoformat()} ({time_zone})"
        weather_data = f"""Current Time: {weather_time}\nTemperature: {temperature}F\nWeather: {weather_main}\nUV Index: {uvi}\nSunrise Time: {sunrise.isoformat()}\nSunset Time: {sunset.isoformat()}\nRecommendation: {in_or_out}, {day_or_night}"""

    except (requests.RequestException, KeyError) as error:
        logging.error("Error fetching weather data: %s", error)

    return {"weatherMessage": weather_message, "weatherData": weather_data, "weatherTime": weather_time}


if __name__ == "__main__":
    date_time = "2024-11-21T10:30:00+00:00"
    city_name = "San Francisco"
    weather_info = get_weather(date_time, city_name)
    logging.info(weather_info)
