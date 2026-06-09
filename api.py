from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np
import math
import requests
from datetime import datetime

app = FastAPI(title="Eco-Twin Smart Microgrid API")

# 1. Load the trained brain
try:
    model = PPO.load("ppo_smart_microgrid")
    print("AI Model loaded successfully. Ready for IoT connections.")
except Exception as e:
    print("WARNING: 'ppo_smart_microgrid.zip' not found. Did you run train_ai.py?")

# 2. Hardware Data Contract (The ESP32 only sends these two numbers)
class SensorData(BaseModel):
    battery_soc: float           
    campus_demand: float         

# 3. Live Weather Function (Hardcoded for Mohammedia, Morocco)
def get_live_solar_forecast(lat=33.69, lon=-7.38):
    """Fetches live cloud cover for Mohammedia and converts it to a 4-hour solar forecast."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=cloudcover&forecast_hours=4"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        cloud_data = data['hourly']['cloudcover']
        solar_array = []
        current_hour = datetime.now().hour

        for i, cloud_percent in enumerate(cloud_data):
            pred_hour = (current_hour + i) % 24

            # Nighttime safety override (7 PM to 5 AM = 0.0 Solar)
            if pred_hour > 18 or pred_hour < 6:
                solar_array.append(0.0)
            else:
                # Convert cloud percentage to solar efficiency
                solar_efficiency = 1.0 - (cloud_percent / 100.0)
                solar_array.append(round(solar_efficiency, 2))

        return solar_array
    except Exception as e:
        print(f"Weather API Error: {e}")
        # Failsafe: Assume no sun if the internet drops
        return [0.0, 0.0, 0.0, 0.0]

# 4. The Main Execution Loop
@app.post("/predict_action")
def generate_action(data: SensorData):
    # Get current time and live Mohammedia weather
    current_hour = datetime.now().hour
    live_solar_forecast = get_live_solar_forecast()

    # Time Encoding Math
    hour_sin = math.sin(2 * math.pi * current_hour / 24.0)
    hour_cos = math.cos(2 * math.pi * current_hour / 24.0)

    # Build the 8-variable State Vector
    obs = np.array([
        data.battery_soc, 
        live_solar_forecast[0], # Now
        live_solar_forecast[1], # +1 hour
        live_solar_forecast[2], # +2 hours
        live_solar_forecast[3], # +3 hours
        data.campus_demand, 
        hour_sin, 
        hour_cos
    ], dtype=np.float32)

    # Ask the RL Brain for a decision
    action, _ = model.predict(obs, deterministic=True)
    ai_command = float(action[0]) 

    # Translate to IoT physical relay commands
    if ai_command < -0.1:
        status = "DISCHARGE"
    elif ai_command > 0.1:
        status = "CHARGE"
    else:
        status = "IDLE"

    return {
        "hardware_command": status,
        "solar_forecast_used": live_solar_forecast,
        "ai_raw_value": round(ai_command, 3)
    }
