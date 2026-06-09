import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class MicroGridEnv(gym.Env):
    def __init__(self):
        super(MicroGridEnv, self).__init__()

        # --- HARDWARE LIMITS ---
        self.battery_capacity_kwh = 45.0  
        self.max_power_kw = 5.0           
        self.efficiency = 0.95            
        self.min_soc = 0.20               
        self.max_solar_kw = 37.0          
        self.max_demand_kw = 36.0        

        # --- SIMULATION SETTINGS ---
        self.current_step = 0
        self.max_steps = 24 * 30  

        # --- AI BOUNDARIES ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state_soc = 0.5 
        return self._get_observation(), {}

    def step(self, action):
        ai_action = float(action[0]) 
        requested_power_kw = ai_action * self.max_power_kw 

        if requested_power_kw > 0: 
            energy_change = (requested_power_kw * self.efficiency) / self.battery_capacity_kwh
        else:                      
            energy_change = (requested_power_kw / self.efficiency) / self.battery_capacity_kwh

        new_soc = np.clip(self.state_soc + energy_change, 0.0, 1.0)
        actual_power_kw = (new_soc - self.state_soc) * self.battery_capacity_kwh
        self.state_soc = new_soc

        current_solar_kw = self._get_solar_data(self.current_step) * self.max_solar_kw 
        current_demand_kw = self._get_demand_data(self.current_step) * self.max_demand_kw
        grid_power_needed = current_demand_kw - current_solar_kw + actual_power_kw

        reward = 0.0
        if grid_power_needed > 0: reward -= grid_power_needed * 0.15 
        if self.state_soc < self.min_soc: reward -= 100.0 * (self.min_soc - self.state_soc) 

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        hour_sin, hour_cos = self._get_time_encoding(self.current_step)
        obs = np.array([
            self.state_soc,
            self._get_solar_data(self.current_step),
            self._get_solar_data(self.current_step + 1), 
            self._get_solar_data(self.current_step + 2), 
            self._get_solar_data(self.current_step + 3), 
            self._get_demand_data(self.current_step),
            hour_sin, hour_cos
        ], dtype=np.float32)
        return obs

    def _get_solar_data(self, step):
        # Fake bell-curve weather for high-speed training
        hour = step % 24
        if 6 <= hour <= 18: return math.sin((hour - 6) * math.pi / 12)
        return 0.0 

    def _get_demand_data(self, step):
        hour = step % 24
        profile = [0.2]*5 + [0.3, 0.5, 0.8, 1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.8, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]
        return profile[hour]

    def _get_time_encoding(self, step):
        hour = step % 24
        return math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)
