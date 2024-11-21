from datetime import datetime


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
