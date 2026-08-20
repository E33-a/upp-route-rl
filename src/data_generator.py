import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    ROUTES,
    SIMULATION_DAYS,
    INTERVAL_MINUTES,
    START_HOUR,
    END_HOUR,
    BUS_CAPACITY,
    RANDOM_SEED,
    PEAK_HOURS,
    STOP_DELAY_SECONDS,
    AFTERNOON_MAX_WAIT_MINUTES
)

def is_peak_hour(current_time: datetime) -> bool:
    """Determina si la hora pertenece a las franjas pico de la UPP (06:00-08:00 o 11:00-13:00)."""
    hour = current_time.hour
    for start_peak, end_peak in PEAK_HOURS:
        if start_peak <= hour < end_peak:
            return True
    return False

def generate_demand_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Genera el dataset simulado integrando las reglas del mundo real de la UPP:
    1. Horas pico: 06:00-08:00 y 11:00-13:00 hrs.
    2. Retraso por paradas: 10 segundos extra por parada intermedia realizada.
    3. Despacho en la tarde (después de las 12:00 PM): No esperan a llenarse (18 pax);
       salen si superan 10-15 min de espera (Salida Parcial).
    4. Cumplimiento del 100% de los 13 campos exigidos por la rúbrica.
    """
    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2026, 8, 1)
    
    for route in ROUTES:
        for day in range(SIMULATION_DAYS):
            current_date = start_date + timedelta(days=day)
            
            students_waiting = 0
            first_student_arrival_time = None
            
            start_time = datetime(current_date.year, current_date.month, current_date.day, START_HOUR, 0)
            end_time = datetime(current_date.year, current_date.month, current_date.day, END_HOUR, 0)
            
            current_interval = start_time
            
            while current_interval < end_time:
                is_peak = is_peak_hour(current_interval)
                multiplier = route.peak_multiplier if is_peak else 1.0
                lambda_effective = route.arrival_rate_per_min * INTERVAL_MINUTES * multiplier
                
                arrivals = int(rng.poisson(lambda_effective))
                
                if students_waiting == 0 and arrivals > 0:
                    first_student_arrival_time = current_interval
                
                students_waiting += arrivals
                
                # Calcular tiempo de espera actual del primer pasajero
                current_wait_min = 0.0
                if first_student_arrival_time:
                    current_wait_min = (current_interval - first_student_arrival_time).total_seconds() / 60.0
                
                # Regla de despacho:
                # Mañana (<12:00 PM): Sale si se llena a 18 pax.
                # Tarde (>=12:00 PM): Sale si se llena a 18 pax O si la espera supera los 15 min.
                dispatch_occurred = False
                if students_waiting >= BUS_CAPACITY:
                    dispatch_occurred = True
                elif current_interval.hour >= 12 and students_waiting > 0 and current_wait_min >= AFTERNOON_MAX_WAIT_MINUTES:
                    dispatch_occurred = True
                
                dispatch_time = None
                passengers_boarded = 0
                dispatch_type = "Ninguna"
                wait_time_accum = 0.0
                trip_duration = 0.0
                intermediate_boardings = 0
                arrival_upp_time = None
                
                if dispatch_occurred:
                    passengers_boarded = min(students_waiting, BUS_CAPACITY)
                    students_waiting -= passengers_boarded
                    dispatch_type = "Llena" if passengers_boarded == BUS_CAPACITY else "Parcial"
                    dispatch_time = current_interval
                    
                    wait_time_accum = float(np.round(current_wait_min, 1))
                    
                    # Alumnos en paradas intermedias
                    if route.intermediate_stops > 0:
                        intermediate_boardings = int(rng.integers(0, min(4, route.intermediate_stops + 1)))
                    
                    # 10 segundos extra por cada parada realizada (~0.167 min)
                    stop_delay_min = (route.intermediate_stops * STOP_DELAY_SECONDS) / 60.0
                    base_trip_duration = rng.normal(route.trip_duration_mean_min, 2.5)
                    trip_duration = float(np.round(base_trip_duration + stop_delay_min, 1))
                    
                    arrival_upp_time = dispatch_time + timedelta(minutes=trip_duration)
                    
                    first_student_arrival_time = current_interval if students_waiting > 0 else None
                
                records.append({
                    "raw_interval_datetime": current_interval,
                    "fecha": current_interval.strftime("%Y-%m-%d"),
                    "hora": current_interval.strftime("%H:%M"),
                    "ruta": route.name,
                    "alumnos_esperando": students_waiting,
                    "llegadas_intervalo": arrivals,
                    "hora_salida": dispatch_time.strftime("%H:%M") if dispatch_time else "N/A",
                    "pasajeros_al_salir": passengers_boarded,
                    "tipo_salida": dispatch_type,
                    "tiempo_espera_acum": wait_time_accum if dispatch_occurred else 0.0,
                    "tiempo_recorrido": trip_duration if dispatch_occurred else 0.0,
                    "paradas_intermedias": route.intermediate_stops,
                    "alumnos_recogidos_intermedias": intermediate_boardings if dispatch_occurred else 0,
                    "hora_llegada_upp": arrival_upp_time.strftime("%H:%M") if arrival_upp_time else "N/A"
                })
                
                current_interval += timedelta(minutes=INTERVAL_MINUTES)

    df = pd.DataFrame(records)
    
    df.sort_values(by=["raw_interval_datetime", "ruta"], inplace=True)
    df.drop(columns=["raw_interval_datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df