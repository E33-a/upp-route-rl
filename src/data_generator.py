import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    ROUTES,
    SIMULATION_DAYS,
    START_HOUR,
    END_HOUR,
    BUS_CAPACITY,
    RANDOM_SEED,
    PEAK_HOURS,
    STOP_DELAY_SECONDS,
    AFTERNOON_MAX_WAIT_MINUTES
)

def is_peak_hour(current_time: datetime) -> bool:
    """Determines whether current_time falls within UPP peak hours (06:00-08:00 or 11:00-13:00)."""
    hour = current_time.hour
    for start_peak, end_peak in PEAK_HOURS:
        if start_peak <= hour < end_peak:
            return True
    return False

def generate_demand_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates event-driven van dispatch operational dataset for 21 days (3 weeks).
    Each row represents an actual van dispatch event (NO 'N/A' values).
    
    Real-world UPP business rules:
    1. Sequential Combi Queueing: ONLY ONE combi loads at a time per route. A new combi
       cannot start loading until the previous combi departs. Leftover queue passengers carry over.
    2. Before 12:00 PM (Morning): Vans STRICTLY wait until full (18 passengers). 'tipo_salida' is always 'Llena'.
    3. After 12:00 PM (Afternoon): Vans depart when full (18 pax) OR when max wait time (15 min) is reached.
       'tipo_salida' can be 'Llena' or 'Parcial'.
    4. Peak hours (06:00-08:00 and 11:00-13:00) feature higher arrival rates.
    5. Trip duration includes +10 seconds delay per intermediate stop.
    6. 100% complete fields: hora_salida and hora_llegada_upp are fully populated in all rows.
    """
    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2026, 8, 1)
    
    for route in ROUTES:
        for day in range(SIMULATION_DAYS):
            current_date = start_date + timedelta(days=day)
            
            day_start = datetime(current_date.year, current_date.month, current_date.day, START_HOUR, 0)
            day_end = datetime(current_date.year, current_date.month, current_date.day, END_HOUR, 0)
            
            sim_time = day_start
            leftover_queue = 0
            
            while sim_time < day_end:
                # Combi queueing: Start with leftover passengers from previous combi
                students_waiting = leftover_queue
                leftover_queue = 0
                
                first_arrival_time = sim_time
                current_time = sim_time
                total_arrivals_during_wait = 0
                
                dispatch_occurred = False
                
                # Simulate minute-by-minute student arrivals until dispatch trigger
                while current_time < day_end and not dispatch_occurred:
                    is_peak = is_peak_hour(current_time)
                    multiplier = route.peak_multiplier if is_peak else 1.0
                    lambda_effective = route.arrival_rate_per_min * multiplier
                    
                    # Arrivals per minute (Poisson)
                    minute_arrivals = int(rng.poisson(lambda_effective))
                    
                    if students_waiting == 0 and minute_arrivals > 0:
                        first_arrival_time = current_time
                    
                    students_waiting += minute_arrivals
                    total_arrivals_during_wait += minute_arrivals
                    
                    wait_minutes = (current_time - first_arrival_time).total_seconds() / 60.0 if students_waiting > 0 else 0.0
                    
                    # Dispatch Rules:
                    # Morning (< 12:00 PM): STRICTLY wait until full (18 pax)
                    # Afternoon (>= 12:00 PM): Wait until full (18 pax) OR max 15 minutes wait
                    if current_time.hour < 12:
                        if students_waiting >= BUS_CAPACITY:
                            dispatch_occurred = True
                    else:
                        if students_waiting >= BUS_CAPACITY:
                            dispatch_occurred = True
                        elif students_waiting > 0 and wait_minutes >= AFTERNOON_MAX_WAIT_MINUTES:
                            dispatch_occurred = True
                    
                    if not dispatch_occurred:
                        current_time += timedelta(minutes=1)
                
                if dispatch_occurred:
                    passengers_boarded = min(students_waiting, BUS_CAPACITY)
                    leftover_queue = students_waiting - passengers_boarded
                    dispatch_type = "Llena" if passengers_boarded == BUS_CAPACITY else "Parcial"
                    
                    dispatch_time = current_time
                    wait_time_accum = float(np.round((dispatch_time - first_arrival_time).total_seconds() / 60.0, 1))
                    
                    # Intermediate stop boardings and delay
                    intermediate_boardings = 0
                    if route.intermediate_stops > 0:
                        intermediate_boardings = int(rng.integers(0, min(4, route.intermediate_stops + 1)))
                    
                    stop_delay_min = (route.intermediate_stops * STOP_DELAY_SECONDS) / 60.0
                    base_trip_duration = rng.normal(route.trip_duration_mean_min, 2.5)
                    trip_duration = float(np.round(max(10.0, base_trip_duration + stop_delay_min), 1))
                    
                    arrival_upp_time = dispatch_time + timedelta(minutes=trip_duration)
                    
                    records.append({
                        "fecha": dispatch_time.strftime("%Y-%m-%d"),
                        "hora": first_arrival_time.strftime("%H:%M"),
                        "ruta": route.name,
                        "alumnos_esperando": passengers_boarded + leftover_queue,
                        "llegadas_intervalo": total_arrivals_during_wait,
                        "hora_salida": dispatch_time.strftime("%H:%M"),
                        "pasajeros_al_salir": passengers_boarded,
                        "tipo_salida": dispatch_type,
                        "tiempo_espera_acum": wait_time_accum,
                        "tiempo_recorrido": trip_duration,
                        "paradas_intermedias": route.intermediate_stops,
                        "alumnos_recogidos_intermedias": intermediate_boardings,
                        "hora_llegada_upp": arrival_upp_time.strftime("%H:%M")
                    })
                    
                    # Advance simulation time to dispatch time for next combi (sequential single loading)
                    sim_time = dispatch_time + timedelta(minutes=1)
                else:
                    # End of operational day reached
                    break

    df = pd.DataFrame(records)
    
    # Sort chronologically by date and departure time
    df['dt_sort'] = pd.to_datetime(df['fecha'] + ' ' + df['hora_salida'])
    df.sort_values(by=['dt_sort', 'ruta'], inplace=True)
    df.drop(columns=['dt_sort'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df