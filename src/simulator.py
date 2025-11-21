import numpy as np
import random

# ==============================================================================
# SIMULATION SETTINGS
# ==============================================================================
DRIVER_CONSISTENCY = 0.20    
PIT_STOP_ERROR_CHANCE = 0.05 
PIT_STOP_ERROR_TIME = 3.0    
FUEL_PENALTY_PER_LAP = 0.035 

def generate_logical_strategies(total_laps):
    strategies = []
    # 1-STOP
    start_window = int(total_laps * 0.25)
    end_window = int(total_laps * 0.45)
    for lap in range(start_window, end_window, 3):
        strategies.append([('SOFT', lap), ('HARD', total_laps - lap)])
        strategies.append([('MEDIUM', lap), ('HARD', total_laps - lap)])
    for lap in range(int(total_laps * 0.6), int(total_laps * 0.8), 4):
        strategies.append([('HARD', lap), ('SOFT', total_laps - lap)])
    # 2-STOP
    for pit1 in range(12, 20, 3):
        remaining = total_laps - pit1
        stint2 = int(remaining * 0.6)
        stint3 = remaining - stint2
        strategies.append([('SOFT', pit1), ('HARD', stint2), ('SOFT', stint3)])
        strategies.append([('SOFT', pit1), ('MEDIUM', stint2), ('SOFT', stint3)])
    return strategies

def simulate_single_race(strategy, total_laps, tire_stats, pit_loss, track_evo=0.0):
    """
    Runs ONE stochastic race simulation.
    Added: track_evo parameter to simulate rubbering in.
    """
    current_time = 0
    current_lap_global = 0
    cumulative_time = [] 
    
    for i, (compound, stint_laps) in enumerate(strategy):
        base, deg = tire_stats.get(compound, (100.0, 0.1))
        
        for tire_age in range(1, stint_laps + 1):
            current_lap_global += 1
            if current_lap_global > total_laps: break
            
            # --- PHYSICS ---
            wear = tire_age * deg
            
            # Car gets lighter -> Faster (-)
            fuel_relief = current_lap_global * FUEL_PENALTY_PER_LAP
            
            # Track gets grippier -> Faster (-)
            track_grip = current_lap_global * track_evo
            
            # Monte Carlo Noise
            noise = np.random.normal(0, DRIVER_CONSISTENCY)
            
            # CALCULATION: Base + Wear - Fuel - TrackEvo + Noise
            lap_time = base + wear - fuel_relief - track_grip + noise
            
            current_time += lap_time
            cumulative_time.append(current_time) 
            
        # Pit Stop
        if i < len(strategy) - 1:
            stop_time = pit_loss
            if random.random() < PIT_STOP_ERROR_CHANCE:
                stop_time += np.random.exponential(PIT_STOP_ERROR_TIME)
            
            current_time += stop_time
            if cumulative_time:
                cumulative_time[-1] += stop_time
            
    return current_time, cumulative_time