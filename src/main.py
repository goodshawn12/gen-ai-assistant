import os
import logging
import requests
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from zoneinfo import ZoneInfo

openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

if openai_client.api_key is None:
    logging.error("OPENAI_API_KEY environment variable is not set")

weather_api_key = os.getenv('REACT_APP_WEATHER_API_KEY')
if weather_api_key is None:
    logging.error("REACT_APP_WEATHER_API_KEY environment variable is not set")

logging.basicConfig(filename='main.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def request_to_openai(prompt: str, model: str="gpt-4o-mini", client=openai_client) -> str:

    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=500,
            temperature=0,
        )
        logging.info(chat_completion)    
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error making API request to OpenAI: {e}")
        return None


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

    return {"is_normal_bp": is_normal_bp, "bp_message": bp_message}


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

        # Extract relevant dat
        data_current = data["data"][0]
        time_zone = data["timezone"]
        current_time = datetime.fromtimestamp(data_current["dt"], tz=ZoneInfo(time_zone))
        sunrise = datetime.fromtimestamp(data_current["sunrise"], tz=ZoneInfo(time_zone))
        sunset = datetime.fromtimestamp(data_current["sunset"], tz=ZoneInfo(time_zone))
        temperature = data_current["temp"]
        uvi = data_current["uvi"]
        weather_main = data_current["weather"][0]["main"]

        # Assess outdoor environment
        in_or_out, day_or_night, _, _ = assess_outdoor_env(temperature, weather_main, current_time, sunrise, sunset).values()

        # Construct weather message and data
        weather_message = f"{in_or_out}, {day_or_night}"
        weather_time = f"{current_time.strftime("%Y-%m-%d %H:%M:%S")} ({time_zone})"
        weather_data = f"Current Time: {weather_time}\n Temperature: {temperature}F\n Weather: {weather_main}\n UV Index: {uvi}\n Sunrise Time: {sunrise.strftime("%H:%M:%S")}\n Sunset Time: {sunset.strftime("%H:%M:%S")}\n Recommendation: {in_or_out}, {day_or_night}"

    except (requests.RequestException, KeyError) as error:
        logging.error("Error fetching weather data: %s", error)

    return {"weather_message": weather_message, "weather_data": weather_data, "weather_time": weather_time}


def run_ai_pipeline_single_request(input_dict: dict):

    # Extract values from the input dictionary
    patient_name = input_dict.get("patientName")
    patient_sex = input_dict.get("patientSex")
    patient_age = input_dict.get("patientAge")
    patient_act = input_dict.get("patientAct")
    data_sbp = input_dict.get("data_sbp")
    data_dbp = input_dict.get("data_dbp")
    location_city_name = input_dict.get("location_city_name")
    record_data_time = input_dict.get("record_data_time")
    history_bp = input_dict.get("history_bp")
    history_message = input_dict.get("history_message")

    patient_info = f"Sex: {patient_sex}; Age: {patient_age};"
    header_message = f"Dear {patient_name}, "

    # TODO: assume blood pressure is integer, to accommodate float in the future        
    dict_bp = assess_bp(int(data_sbp), int(data_dbp))
    logging.info(dict_bp)

    dict_weather = get_weather(record_data_time, location_city_name)
    logging.info(dict_weather)

    llm_on_history = "Sample LLM response on BP history. "
    if history_bp != "":
        bp_history = f"{history_bp}\n{dict_weather.get("weather_time")}, {data_sbp}, {data_dbp};"

        # # Call the Bedrock API with the necessary parameters
        # response = await amplify_client.queries.ask_bedrock({
        #     "patientMessage": bp_history,
        #     "weatherMessage": "",
        #     "activityMessage": "",
        #     "conditionMessage": "history"
        # })

        # data = response.get("data")
        # errors = response.get("errors")

        # if not errors:
        #     llm_message_data = f"{data.get('body', '')}\n" if data else ""
        #     logging.info(llm_message_data)
        # else:
        #     logging.error(errors)
    else:
        bp_history = f"{dict_weather.get("weather_time")}, {data_sbp}, {data_dbp};"
        logging.warning("No data history available.")

    logging.info(bp_history)

    llm_on_recommendation = "Sample LLM response on recommendation"
    is_normal_bp = dict_bp.get("is_normal_bp")
    if is_normal_bp == 1:
        logging.debug("is_normal_bp: True")

        # # Call the Bedrock API with necessary parameters
        # response = await amplify_client.queries.ask_bedrock({
        #     "patientMessage": patient_info,
        #     "weatherMessage": weather_message,
        #     "activityMessage": patient_act,
        #     "conditionMessage": "recommendation"
        # })

        # data = response.get("data")
        # errors = response.get("errors")

        # if not errors:
        #     llm_message_rec = data.get("body", "")
        # else:
        #     logging.error(errors)

    # Construct the output message
    output_message = header_message + dict_bp.get("bp_message") + llm_on_history + llm_on_recommendation
    logging.info(output_message)

    # Construct the history entry
    history_heading = f"{dict_weather.get("weather_time")}, SBP: {data_sbp}, DBP: {data_dbp}\n"
    if not history_message:  # empty string
        history_message = history_heading + output_message + "\n"
    else:
        history_message = history_message + "\n" + history_heading + output_message + "\n"
    
    logging.info(history_message)


if __name__ == "__main__":

    # TODO: load below data from file
    input_dict = {
        "patientName": "First, Last",
        "patientSex": "F",
        "patientAge": "65",
        "patientAct": "exercise",
        "data_sbp": "120",
        "data_dbp": "80",
        "location_city_name": "San Diego",
        "record_data_time": "2024-11-21 10:30:00-08",
        "history_bp": "",
        "history_message": "Test History Message"
    }

    # run_ai_pipeline_single_request(input_dict)
    prompt = "You are a marraige relationship builder and adventurer, what are some ideas for things to do to celebrate my wife and my 6th marriage anniversary?"
    response = request_to_openai(prompt=prompt)
    logging.info(response)
