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
    PEAK_HOURS
)

def is_peak_hour(current_time: datetime) -> bool:
    """Determina si la hora dada pertenece a una franja pico."""
    hour = current_time.hour
    for start_peak, end_peak in PEAK_HOURS:
        if start_peak <= hour < end_peak:
            return True
    return False

def get_current_arrival_rate(current_time: datetime, route) -> float:
    """Retorna la tasa de llegada (alumnos/minuto) según el horario."""
    multiplier = route.peak_multiplier if is_peak_hour(current_time) else 1.0
    return route.arrival_rate_per_min * multiplier

def generate_demand_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Genera un dataset estocástico basado en EVENTOS DE DESPACHO (viajes reales).
    Cada fila representa una combi que se llenó y partió hacia la UPP.
    Los registros se devuelven ordenados cronológicamente por fecha y hora de salida.
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
            
            while sim_time < day_end:
                arrival_times = []
                current_time = sim_time
                
                # Simular la llegada consecutiva de 18 alumnos mediante distribución Exponencial
                for _ in range(BUS_CAPACITY):
                    rate = get_current_arrival_rate(current_time, route)
                    inter_arrival = rng.exponential(scale=1.0 / rate)
                    current_time += timedelta(minutes=inter_arrival)
                    arrival_times.append(current_time)
                
                dispatch_time = arrival_times[-1]
                
                if dispatch_time >= day_end:
                    break
                
                first_arrival = arrival_times[0]
                
                # Tiempos de espera
                wait_max_min = (dispatch_time - first_arrival).total_seconds() / 60.0
                wait_times = [(dispatch_time - arr).total_seconds() / 60.0 for arr in arrival_times]
                wait_avg_min = float(np.mean(wait_times))
                
                # Recorrido a la UPP (Normal)
                trip_duration = float(np.round(rng.normal(route.trip_duration_mean_min, 2.5), 1))
                arrival_upp_time = dispatch_time + timedelta(minutes=trip_duration)
                
                peak_flag = 1 if is_peak_hour(dispatch_time) else 0
                
                records.append({
                    "raw_dispatch_datetime": dispatch_time,
                    "fecha": dispatch_time.strftime("%Y-%m-%d"),
                    "ruta": route.name,
                    "hora_inicio_espera": first_arrival.strftime("%H:%M"),
                    "hora_salida": dispatch_time.strftime("%H:%M"),
                    "pasajeros": BUS_CAPACITY,
                    "tiempo_espera_max_min": float(np.round(wait_max_min, 1)),
                    "tiempo_espera_prom_min": float(np.round(wait_avg_min, 1)),
                    "es_hora_pico": peak_flag,
                    "tiempo_recorrido_min": trip_duration,
                    "hora_llegada_upp": arrival_upp_time.strftime("%H:%M")
                })
                
                sim_time = dispatch_time

    df = pd.DataFrame(records)
    
    # Ordenar cronológicamente por estampa de tiempo real
    df = df.sort_values(by="raw_dispatch_datetime").reset_index(drop=True)
    df.drop(columns=["raw_dispatch_datetime"], inplace=True)
    
    # Asignar id_viaje correlativo cronológico (1, 2, 3...)
    df.insert(0, "id_viaje", range(1, len(df) + 1))
    
    return df