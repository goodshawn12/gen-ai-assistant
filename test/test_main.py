import unittest
from datetime import datetime, timedelta
from src.main import assess_bp, assess_outdoor_env


class TestFunctions(unittest.TestCase):
    def test_assess_bp(self):
        # Test cases for assess_bp
        self.assertEqual(
            assess_bp(120, 80),
            {"isNormalBP": 1, "bpMessage": "Thanks for sharing your reading! Your blood pressure looks great.\n"}
        )
        self.assertEqual(
            assess_bp(85, 45),
            {
                "isNormalBP": 0,
                "bpMessage": (
                    "Thanks for sharing your reading! Your blood pressure reading is abnormal today. "
                    "If you haven't done so, could you recheck your blood pressure to ensure the reading is accurate?\n"
                ),
            }
        )
        self.assertEqual(
            assess_bp(140, 95),
            {
                "isNormalBP": 0,
                "bpMessage": (
                    "Thanks for sharing your reading! Your blood pressure is higher than normal today. "
                    "Watch for the following symptoms such as dizziness, headache, and chest discomfort. "
                    "Contact your provider if needed. Otherwise, recheck your blood pressure after a few minutes of rest.\n"
                ),
            }
        )

    def test_assess_outdoor_env(self):
        # Test cases for assess_outdoor_env
        sunrise = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        sunset = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)

        current_time_day = sunrise + timedelta(hours=6)  # 12 PM
        current_time_night = sunset + timedelta(hours=1)  # 7 PM

        self.assertEqual(
            assess_outdoor_env(70, "Clear", current_time_day, sunrise, sunset),
            {
                "inOrOut": "outdoor",
                "dayOrNight": "daytime",
                "isGoodTemp": True,
                "isRain": False,
            }
        )
        self.assertEqual(
            assess_outdoor_env(40, "Rainy", current_time_day, sunrise, sunset),
            {
                "inOrOut": "indoor",
                "dayOrNight": "daytime",
                "isGoodTemp": False,
                "isRain": True,
            }
        )
        self.assertEqual(
            assess_outdoor_env(80, "Clear", current_time_night, sunrise, sunset),
            {
                "inOrOut": "indoor",
                "dayOrNight": "nighttime",
                "isGoodTemp": True,
                "isRain": False,
            }
        )

if __name__ == "__main__":
    unittest.main()
