import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    ROUTES,
    SIMULATION_DAYS,
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
    
    Real-world UPP operational rules:
    1. Intermediate Stop Pickups:
       - Full combis ('Llena') pick up up to 8 standing passengers along intermediate stops (0-8 pax).
       - Partial combis in transit ('Parcial') can pick up up to 9-10 passengers (0-10 pax).
    2. New Field: 'total_pasajeros_final' = pasajeros_al_salir + alumnos_recogidos_intermedias.
    3. Base vs Transit Operations:
       - Las Vías: Makes base until 13:00 hrs. After 13:00 hrs, vans pass in transit until ~18:30 hrs.
       - Puente Tuzos: Makes base until 12:00 hrs. After 12:00 hrs, vans pass in transit until ~14:00 hrs.
    4. Strict Time Sorting: All dispatches ordered strictly by departure date and time across routes.
    5. Dynamic Intermediate Stops: 0 to 4 stops per trip (+10s delay per stop made).
    6. Sequential Combi Queueing: Single active loading combi per route with leftover queue carryover.
    """
    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2026, 8, 1)
    
    for route in ROUTES:
        for day in range(SIMULATION_DAYS):
            current_date = start_date + timedelta(days=day)
            
            start_hour_int = int(route.base_start_hour)
            start_min_int = int((route.base_start_hour - start_hour_int) * 60)
            
            end_hour_int = int(route.last_departure_hour)
            end_min_int = int((route.last_departure_hour - end_hour_int) * 60)
            
            day_start = datetime(current_date.year, current_date.month, current_date.day, start_hour_int, start_min_int)
            day_end = datetime(current_date.year, current_date.month, current_date.day, end_hour_int, end_min_int)
            
            sim_time = day_start
            leftover_queue = 0
            
            while sim_time < day_end:
                students_waiting = leftover_queue
                leftover_queue = 0
                
                first_arrival_time = sim_time
                current_time = sim_time
                total_arrivals_during_wait = 0
                
                dispatch_occurred = False
                
                # Simulate minute-by-minute arrivals
                while current_time < day_end and not dispatch_occurred:
                    is_peak = is_peak_hour(current_time)
                    multiplier = route.peak_multiplier if is_peak else 1.0
                    lambda_effective = route.arrival_rate_per_min * multiplier
                    
                    minute_arrivals = int(rng.poisson(lambda_effective))
                    
                    if students_waiting == 0 and minute_arrivals > 0:
                        first_arrival_time = current_time
                    
                    students_waiting += minute_arrivals
                    total_arrivals_during_wait += minute_arrivals
                    
                    wait_minutes = (current_time - first_arrival_time).total_seconds() / 60.0 if students_waiting > 0 else 0.0
                    current_hour_decimal = current_time.hour + (current_time.minute / 60.0)
                    
                    # Dispatch Logic:
                    # Base Hours (< base_end_hour): Strictly wait for full 18 pax
                    # Transit Hours (>= base_end_hour): Depart when full (18 pax) OR max 15 min wait
                    if current_hour_decimal < route.base_end_hour:
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
                    
                    # Dynamic intermediate stops (0 to 4 stops made per trip)
                    stops_made = int(rng.integers(0, 5))
                    intermediate_boardings = 0
                    
                    if stops_made > 0:
                        if dispatch_type == "Llena":
                            # Full combi: pick up up to 8 standing passengers along route
                            intermediate_boardings = int(rng.integers(0, 9))
                        else:
                            # Partial combi in transit: can pick up up to 9-10 passengers along route
                            intermediate_boardings = int(rng.integers(0, 11))
                    
                    total_pasajeros_final = passengers_boarded + intermediate_boardings
                    
                    stop_delay_min = (stops_made * STOP_DELAY_SECONDS) / 60.0
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
                        "paradas_intermedias": stops_made,
                        "alumnos_recogidos_intermedias": intermediate_boardings,
                        "total_pasajeros_final": total_pasajeros_final,
                        "hora_llegada_upp": arrival_upp_time.strftime("%H:%M")
                    })
                    
                    sim_time = dispatch_time + timedelta(minutes=1)
                else:
                    break

    df = pd.DataFrame(records)
    
    # Sort STRICTLY by departure date and time regardless of route
    df['dt_sort'] = pd.to_datetime(df['fecha'] + ' ' + df['hora_salida'])
    df.sort_values(by=['dt_sort', 'ruta'], inplace=True)
    df.drop(columns=['dt_sort'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df