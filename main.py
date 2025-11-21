import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import loader, physics, simulator

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
YEAR = 2024
CIRCUIT = 'Bahrain'
HERO_DRIVER = 'VER'
PIT_LOSS = 24       # Adjusted for Red Bull efficiency
N_SIMULATIONS = 70000  
TRACK_EVO_PER_LAP = 0.045 
SPRINT_FUEL_OFFSET = 2.5 

# ==============================================================================
# 2. VISUALIZATION
# ==============================================================================
def plot_pit_window(lap_results, best_lap, stop_number):
    laps = sorted(lap_results.keys())
    times = [np.median(lap_results[l]) for l in laps]
    
    plt.figure(figsize=(10, 5))
    plt.plot(laps, [t - min(times) for t in times], color='#1f77b4', linewidth=2, marker='o')
    plt.axvspan(best_lap, best_lap + 1, color='#2ca02c', alpha=0.2, label='Optimal Window')
    
    plt.title(f'Optimization: Pit Stop #{stop_number} ({HERO_DRIVER})', fontsize=14)
    plt.xlabel('Lap Number')
    plt.ylabel('Time Lost (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f'pit_window_{stop_number}_{HERO_DRIVER}.png')
    print(f"📊 Graph saved as 'pit_window_{stop_number}_{HERO_DRIVER}.png'")
    plt.close()

def plot_prediction_vs_reality(simulated_times, session, driver):
    print(f"\n--- PHASE 4: REALITY CHECK ({driver}) ---")
    laps = session.laps.pick_driver(driver)
    laps = laps[laps['LapNumber'] >= 1]
    
    if laps.empty:
        print("No actual race data found.")
        return

    real_times = laps['LapTime'].dt.total_seconds().cumsum().tolist()
    real_laps = laps['LapNumber'].tolist()
    sim_laps = range(1, len(simulated_times) + 1)
    
    final_real = real_times[-1] if real_times else 0
    final_sim = simulated_times[-1] if simulated_times else 0
    gap = final_sim - final_real
    status = "SLOWER" if gap > 0 else "FASTER"
    
    def fmt(s):
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m):02d}:{int(s):02d}"

    print(f"   Actual Race Time: {fmt(final_real)} (Official)")
    print(f"   Model Race Time:  {fmt(final_sim)} (Optimized)")
    print(f"   Difference:       {abs(gap):.2f}s {status} than reality")
    
    plt.figure(figsize=(12, 7))
    plt.plot(sim_laps, simulated_times, label='Simulated Strategy', color='blue', linestyle='--', linewidth=2)
    plt.plot(real_laps, real_times, label='Actual Race', color='red', linewidth=2, alpha=0.7)
    plt.title(f'Prediction vs Reality: {driver} @ {CIRCUIT} {YEAR}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f'validation_{driver}.png')
    print(f"📊 Validation Graph saved as 'validation_{driver}.png'")
    plt.show()

def compare_strategy_decisions(best_strategy, session, driver):
    print(f"\n--- STRATEGY REPORT CARD ({driver}) ---")
    laps = session.laps.pick_driver(driver)
    real_stints = []
    current_compound = None
    current_start = 1
    
    for i, row in laps.iterrows():
        lap = int(row['LapNumber'])
        compound = row['Compound']
        if compound != current_compound:
            if current_compound is not None:
                real_stints.append((current_compound, lap - current_start))
            current_compound = compound
            current_start = lap
    total_laps = int(session.total_laps)
    real_stints.append((current_compound, total_laps - current_start + 1))
    
    def strat_str(s): return " -> ".join([f"{x[0]}({x[1]})" for x in s])
    print(f"   Actual Strategy: {strat_str(real_stints)}")
    print(f"   Model Strategy:  {strat_str(best_strategy)}")

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    print(f"\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║   F1 STRATEGY PREDICTOR: {HERO_DRIVER} @ {CIRCUIT.upper()} {YEAR}                   ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝\n")

    # --- STEP 1: DATA ---
    race = loader.get_race_data(YEAR, CIRCUIT, 'R')
    total_laps = int(race.total_laps) if race.total_laps else 57
    laps_race_grid = loader.get_clean_laps(race, race.drivers)
    laps_race_hero = loader.get_clean_laps(race, [HERO_DRIVER])
    sprint = loader.get_sprint_data(YEAR, CIRCUIT)
    laps_sprint_hero = loader.get_clean_laps(sprint, [HERO_DRIVER]) if sprint else pd.DataFrame()
    
    # NEW: Get Gaps from FP2/Quali
    gap_data = loader.calculate_compound_deltas(YEAR, CIRCUIT)

    # --- STEP 2: PHYSICS ---
    print(f"--> [2/4] Computing Physics...")
    for df in [laps_race_grid, laps_race_hero]:
        df['Fuel_Adjusted_Lap_Time'] = df.apply(
            lambda x: physics.calculate_fuel_correction(
                x['LapTime'].total_seconds(), x['LapNumber'], total_laps, track_evo=TRACK_EVO_PER_LAP
            ), axis=1
        )

    tire_stats = {}
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    
    for c in compounds:
        _, grid_deg = physics.calculate_tire_stats(laps_race_grid, c)
        hero_base, hero_deg = physics.calculate_tire_stats(laps_race_hero, c)
        source = "RACE"
        
        if len(laps_race_hero[laps_race_hero['Compound'] == c]) > 3:
            # Use 1st Percentile (Ultimate Pace)
            hero_base = laps_race_hero[laps_race_hero['Compound'] == c]['Fuel_Adjusted_Lap_Time'].quantile(0.01)
        
        if not hero_base and not laps_sprint_hero.empty:
            s_laps = laps_sprint_hero[laps_sprint_hero['Compound'] == c]
            if len(s_laps) > 3:
                hero_base = s_laps['LapTime'].dt.total_seconds().quantile(0.01) + SPRINT_FUEL_OFFSET
                hero_deg = 0.1 
                source = "SPRINT"

        if hero_base:
            final_deg = hero_deg if (hero_deg and hero_deg > 0.01) else grid_deg
            if not final_deg: final_deg = 0.12 
            tire_stats[c] = (hero_base, final_deg, source)
        
    # Fill Missing with Dynamic Gaps
    if 'SOFT' in tire_stats:
        s_base, s_deg, _ = tire_stats['SOFT']
        if 'MEDIUM' not in tire_stats:
            tire_stats['MEDIUM'] = (s_base + gap_data['SOFT_TO_MED'], max(0.05, s_deg - 0.03), "EST.")
        if 'HARD' not in tire_stats:
            tire_stats['HARD'] = (s_base + gap_data['SOFT_TO_MED'] + gap_data['MED_TO_HARD'], max(0.02, s_deg - 0.06), "EST.")

    # Physics Enforcer
    if 'SOFT' in tire_stats and 'HARD' in tire_stats:
        s_base, s_deg, _ = tire_stats['SOFT']
        h_base, h_deg, h_src = tire_stats['HARD']
        if h_deg >= s_deg:
            tire_stats['HARD'] = (h_base, s_deg * 0.6, "CORR.")
        if h_base < s_base + 0.5:
            tire_stats['HARD'] = (s_base + 1.0, tire_stats['HARD'][1], "CORR.")

    if 'MEDIUM' in tire_stats:
        m_base = tire_stats['MEDIUM'][0]
        s_base = tire_stats['SOFT'][0]
        h_base = tire_stats['HARD'][0]
        if m_base < s_base or m_base > h_base:
             tire_stats['MEDIUM'] = ((s_base + h_base) / 2, tire_stats['MEDIUM'][1], "CORR.")

    print("\n--- TIRE PERFORMANCE MODEL ---")
    for c in compounds:
        if c in tire_stats:
            base, deg, src = tire_stats[c]
            print(f"  {c:<8} | Base: {base:.2f}s | Deg: +{deg:.3f}s/lap  [{src}]")

    # --- STEP 3: STRATEGY ---
    print(f"\n--- PHASE 2: STRATEGY SELECTION ---")
    strategies = simulator.generate_logical_strategies(total_laps)
    sim_stats = {k: (v[0], v[1]) for k, v in tire_stats.items()}
    
    results = []
    for strategy in strategies:
        if any(stint[0] not in sim_stats for stint in strategy): continue
        times = []
        for _ in range(N_SIMULATIONS):
            t, _ = simulator.simulate_single_race(strategy, total_laps, sim_stats, PIT_LOSS, track_evo=TRACK_EVO_PER_LAP)
            times.append(t)
        results.append((strategy, np.median(times)))

    results.sort(key=lambda x: x[1])
    best_strategy = results[0][0]
    
    def get_strat_name(strat): return " -> ".join([f"{s[0][0]}({s[1]})" for s in strat])
    print(f"\n🏆 BEST STRATEGY: {get_strat_name(best_strategy)}")

    # --- STEP 4: OPTIMIZATION ---
    print(f"\n--- PHASE 3: OPTIMIZATION ---")
    stops_count = len(best_strategy) - 1
    initial_s1 = best_strategy[0][1]
    search_range_1 = range(max(5, initial_s1 - 6), min(total_laps-10, initial_s1 + 6))
    
    res_1 = {}
    for test_lap in search_range_1:
        if stops_count == 1:
            test_strat = [(best_strategy[0][0], test_lap), (best_strategy[1][0], total_laps - test_lap)]
        else:
            stint2_len = best_strategy[1][1]
            stint3_len = total_laps - test_lap - stint2_len
            if stint3_len < 1: continue
            test_strat = [(best_strategy[0][0], test_lap), (best_strategy[1][0], stint2_len), (best_strategy[2][0], stint3_len)]
        times = [simulator.simulate_single_race(test_strat, total_laps, sim_stats, PIT_LOSS, track_evo=TRACK_EVO_PER_LAP)[0] for _ in range(N_SIMULATIONS)]
        res_1[test_lap] = times
    
    best_s1 = min(res_1, key=lambda k: np.median(res_1[k]))
    plot_pit_window(res_1, best_s1, 1)

    best_s2 = 0
    final_strat = []
    if stops_count == 2:
        initial_s2 = best_s1 + best_strategy[1][1]
        search_range_2 = range(max(best_s1 + 5, initial_s2 - 6), min(total_laps - 5, initial_s2 + 6))
        res_2 = {}
        for test_lap_2 in search_range_2:
            stint2_len = test_lap_2 - best_s1
            stint3_len = total_laps - test_lap_2
            test_strat = [(best_strategy[0][0], best_s1), (best_strategy[1][0], stint2_len), (best_strategy[2][0], stint3_len)]
            times = [simulator.simulate_single_race(test_strat, total_laps, sim_stats, PIT_LOSS, track_evo=TRACK_EVO_PER_LAP)[0] for _ in range(N_SIMULATIONS)]
            res_2[test_lap_2] = times
        best_s2 = min(res_2, key=lambda k: np.median(res_2[k]))
        plot_pit_window(res_2, best_s2, 2)
        final_strat = [(best_strategy[0][0], best_s1), (best_strategy[1][0], best_s2 - best_s1), (best_strategy[2][0], total_laps - best_s2)]
    else:
        final_strat = [(best_strategy[0][0], best_s1), (best_strategy[1][0], total_laps - best_s1)]

    print(f"\n✅ FINAL PLAN: {get_strat_name(final_strat)}")

    # --- STEP 5: REALITY CHECK ---
    _, timeline = simulator.simulate_single_race(final_strat, total_laps, sim_stats, PIT_LOSS, track_evo=TRACK_EVO_PER_LAP)
    plot_prediction_vs_reality(timeline, race, HERO_DRIVER)
    compare_strategy_decisions(final_strat, race, HERO_DRIVER)

if __name__ == "__main__":
    main()