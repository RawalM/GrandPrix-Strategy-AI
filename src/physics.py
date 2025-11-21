import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_fuel_correction(lap_time_seconds, lap_number, total_laps, starting_fuel=110, fuel_penalty=0.035, track_evo=0.045):
    """
    Adjusts a lap time to remove the benefits of Fuel Burn AND Track Evolution.
    Returns the 'theoretical' lap time if the car remained heavy and the track remained dirty.
    """
    # 1. FUEL CORRECTION (Car gets faster as it gets lighter)
    # We calculate how much fuel has been burned so far
    fuel_burned_per_lap = starting_fuel / total_laps
    fuel_burned_total = lap_number * fuel_burned_per_lap
    
    # The time GAINED by losing this weight
    fuel_gain = fuel_burned_total * fuel_penalty
    
    # 2. TRACK EVOLUTION CORRECTION (Track gets faster over time)
    # The time GAINED by the track gripping up
    track_gain = lap_number * track_evo
    
    # 3. NORMALIZE
    # We ADD these gains back to the lap time to "slow it down" to baseline conditions
    # Adjusted = Actual + Fuel_Benefit + Track_Benefit
    return lap_time_seconds + fuel_gain + track_gain

def calculate_tire_stats(laps_df, compound):
    """
    Calculates Base Pace (intercept) and Degradation (slope).
    Returns: (base_pace, deg_per_lap)
    """
    compound_laps = laps_df[laps_df['Compound'] == compound].copy()
    
    # Clean data: Remove rows with missing timing info
    clean_laps = compound_laps.dropna(subset=['TyreLife', 'Fuel_Adjusted_Lap_Time'])
    
    # FILTER 1: Remove "Warm-up" Laps
    # The first lap of a stint is often slow due to cold tires or pit exit traffic.
    # Including it ruins the slope calculation.
    clean_laps = clean_laps[clean_laps['TyreLife'] > 1]

    if len(clean_laps) == 0: return None, None
    
    # FILTER 2: Remove Slow Outliers (Yellow Flags / Mistakes)
    # We keep laps that are within 107% of the fastest adjusted lap in this stint
    threshold = clean_laps['Fuel_Adjusted_Lap_Time'].min() * 1.07
    racing_laps = clean_laps[clean_laps['Fuel_Adjusted_Lap_Time'] < threshold]
    
    # We need at least 3 clean racing laps to draw a valid line
    if len(racing_laps) < 3: return None, None

    # LINEAR REGRESSION
    X = racing_laps['TyreLife'].values.reshape(-1, 1)
    y = racing_laps['Fuel_Adjusted_Lap_Time'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    base_pace = model.intercept_[0]
    deg_per_lap = model.coef_[0][0]
    
    return base_pace, deg_per_lap