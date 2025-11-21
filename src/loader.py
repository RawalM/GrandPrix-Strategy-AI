import fastf1
import pandas as pd
import os

# Define Cache Directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/cache')

def setup_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)

def get_race_data(year, circuit, session_type='R'):
    """Loads the session data."""
    setup_cache()
    print(f"--> [Loader] Downloading {session_type} data for {circuit} {year}...")
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"    Error loading {session_type}: {e}")
        return None

def get_clean_laps(session, driver_list=None):
    """Returns clean, green-flag racing laps."""
    if driver_list is None:
        driver_list = session.drivers
    
    # Filter for drivers and accurate laps
    # We only want Green Flag (Status '1') to avoid Safety Car laps
    laps = session.laps.pick_drivers(driver_list).pick_quicklaps().pick_track_status('1')
    
    return laps[['LapNumber', 'Stint', 'Compound', 'TyreLife', 'LapTime', 'Driver']].copy()

def get_sprint_data(year, circuit):
    """Checks for Sprint Data."""
    setup_cache()
    try:
        sprint = fastf1.get_session(year, circuit, 'S')
        sprint.load()
        return sprint
    except:
        return None

def calculate_compound_deltas(year, circuit):
    """
    Determines tire speed gaps using FP2 (Best) or Quali (Backup).
    """
    setup_cache()
    print(f"--> [Loader] Calculating tire performance gaps...")
    
    # 1. Try FP2 (The "Gold Standard" for tire comparison)
    # In FP2, teams do back-to-back runs on different tires in similar conditions.
    try:
        print("    Checking FP2 data (Ideal source)...")
        session = fastf1.get_session(year, circuit, 'FP2')
        session.load()
        laps = session.laps.pick_quicklaps()
    except:
        print("    FP2 unavailable (Sprint weekend?). Switching to Qualifying...")
        try:
            session = fastf1.get_session(year, circuit, 'Q')
            session.load()
            laps = session.laps.pick_quicklaps()
        except:
            print("    No data found. Using Default Estimates.")
            return {'SOFT_TO_MED': 0.8, 'MED_TO_HARD': 0.6}

    # 2. Calculate Gaps
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    best_times = {}
    
    for c in compounds:
        c_laps = laps[laps['Compound'] == c]
        if len(c_laps) > 0:
            # Use the absolute fastest lap seen on this tire
            best_times[c] = c_laps['LapTime'].min().total_seconds()
    
    # 3. Compute Deltas
    gaps = {}
    
    # Gap: Soft -> Medium
    if 'SOFT' in best_times and 'MEDIUM' in best_times:
        diff = best_times['MEDIUM'] - best_times['SOFT']
        # Sanity Check: Gap should be positive (Soft faster) and reasonable (0.1s to 2.5s)
        if 0.1 < diff < 2.5:
            gaps['SOFT_TO_MED'] = diff
        else:
            gaps['SOFT_TO_MED'] = 0.8 # Fallback if data is weird
    else:
        gaps['SOFT_TO_MED'] = 0.8 # Default
        
    # Gap: Medium -> Hard
    if 'MEDIUM' in best_times and 'HARD' in best_times:
        diff = best_times['HARD'] - best_times['MEDIUM']
        if 0.1 < diff < 2.5:
            gaps['MED_TO_HARD'] = diff
        else:
            gaps['MED_TO_HARD'] = 0.6
    else:
        gaps['MED_TO_HARD'] = 0.6 # Default
        
    print(f"    Result: S->M (+{gaps['SOFT_TO_MED']:.3f}s) | M->H (+{gaps['MED_TO_HARD']:.3f}s)")
    return gaps