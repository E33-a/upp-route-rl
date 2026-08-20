from pathlib import Path 
from dataclasses import dataclass 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = BASE_DIR / 'outputs'
PLOTS_DIR = OUTPUTS_DIR / 'plots'

DATA_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

@dataclass(frozen=True)
class RouteConfig:
    name: str
    arrival_rate_per_min: float
    trip_duration_mean_min: float
    peak_multiplier: float
    intermediate_stops: int
    base_start_hour: float
    base_end_hour: float
    last_departure_hour: float

# Configuración de las rutas de combis UPP
ROUTES = [
    RouteConfig(
        name="Las vías",
        arrival_rate_per_min=1.0 / 3.0,
        trip_duration_mean_min=28.0,
        peak_multiplier=1.8,
        intermediate_stops=3,
        base_start_hour=6.0,
        base_end_hour=13.0,          # Hacen base hasta las 13:00 hrs
        last_departure_hour=18.5       # Última combi sale aprox 18:30 hrs
    ),
    RouteConfig(
        name="Puente Tuzos",
        arrival_rate_per_min=1.0 / 7.0,
        trip_duration_mean_min=22.0,
        peak_multiplier=1.5,
        intermediate_stops=2,
        base_start_hour=6.0,
        base_end_hour=12.0,          # Hacen base hasta las 12:00 hrs (luego solo pasan)
        last_departure_hour=14.0       # Pasan en tránsito en la tarde
    )
]

SIMULATION_DAYS = 21
INTERVAL_MINUTES = 15
START_HOUR = 6
END_HOUR = 20
BUS_CAPACITY = 18
RANDOM_SEED = 42
PEAK_HOURS = [(6, 8), (11, 13)]
STOP_DELAY_SECONDS = 10.0  # 10 segundos extra por parada intermedia
AFTERNOON_MAX_WAIT_MINUTES = 15.0  # Después de las 12:00 PM, tiempo máximo de espera de 15 min