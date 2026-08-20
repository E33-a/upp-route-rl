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
    PEAK_HOURS
)

def is_peak_hour(current_time: datetime) -> bool:
    """Determina si la hora dada pertenece a una franja pico."""
    hour = current_time.hour
    for start_peak, end_peak in PEAK_HOURS:
        if start_peak <= hour < end_peak:
            return True
    return False

def generate_demand_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Genera el dataset estocástico oficial cumpliendo al 100% con los 13 campos requeridos:
    
    1. Datos de Demanda:
       - fecha, hora, ruta, alumnos_esperando, llegadas_intervalo
    2. Datos de Despacho:
       - hora_salida, pasajeros_al_salir, tipo_salida, tiempo_espera_acum
    3. Datos de Ruta:
       - tiempo_recorrido, paradas_intermedias, alumnos_recogidos_intermedias, hora_llegada_upp
    """
    rng = np.random.default_rng(seed)
    records = []
    start_date = datetime(2026, 8, 1)
    
    for route in ROUTES:
        for day in range(SIMULATION_DAYS):
            current_date = start_date + timedelta(days=day)
            
            # Estado acumulado del paradero para el día
            students_waiting = 0
            first_student_arrival_time = None
            
            start_time = datetime(current_date.year, current_date.month, current_date.day, START_HOUR, 0)
            end_time = datetime(current_date.year, current_date.month, current_date.day, END_HOUR, 0)
            
            current_interval = start_time
            
            while current_interval < end_time:
                # 1. Tasa de llegada y proceso de Poisson
                is_peak = is_peak_hour(current_interval)
                multiplier = route.peak_multiplier if is_peak else 1.0
                lambda_effective = route.arrival_rate_per_min * INTERVAL_MINUTES * multiplier
                
                arrivals = int(rng.poisson(lambda_effective))
                
                # Registrar tiempo de arribo del primer estudiante si la parada estaba vacía
                if students_waiting == 0 and arrivals > 0:
                    first_student_arrival_time = current_interval
                
                students_waiting += arrivals
                
                # 2. Evaluación de despacho (Política tradicional: Combi Llena = 18 pax)
                dispatch_occurred = False
                dispatch_time = None
                passengers_boarded = 0
                dispatch_type = "Ninguna"
                wait_time_accum = 0.0
                trip_duration = 0.0
                intermediate_boardings = 0
                arrival_upp_time = None
                
                if students_waiting >= BUS_CAPACITY:
                    dispatch_occurred = True
                    passengers_boarded = BUS_CAPACITY
                    students_waiting -= BUS_CAPACITY
                    dispatch_type = "Llena"
                    dispatch_time = current_interval
                    
                    if first_student_arrival_time:
                        wait_time_accum = float(np.round((dispatch_time - first_student_arrival_time).total_seconds() / 60.0, 1))
                    
                    # Alumnos recogidos en paradas intermedias (0 a 3 alumnos si hay espacio/paradas)
                    if route.intermediate_stops > 0:
                        intermediate_boardings = int(rng.integers(0, min(4, route.intermediate_stops + 1)))
                    
                    # Tiempo de recorrido a la UPP (Normal)
                    trip_duration = float(np.round(rng.normal(route.trip_duration_mean_min, 2.5), 1))
                    arrival_upp_time = dispatch_time + timedelta(minutes=trip_duration)
                    
                    # Reset de tiempo para el siguiente grupo si aún quedan alumnos
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
    
    # Ordenar cronológicamente por tiempo real del intervalo
    df.sort_values(by=["raw_interval_datetime", "ruta"], inplace=True)
    df.drop(columns=["raw_interval_datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df