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
    peak_multiplier: float
    trip_duration_mean_min: float
    intermediate_stops: int

ROUTE_PUENTE_TUZOS = RouteConfig(
    name = 'Puente Tuzos',
    arrival_rate_per_min = 1.0 / 7.0,
    peak_multiplier = 1.6,
    trip_duration_mean_min = 20.0,
    intermediate_stops = 2
)

ROUTE_CEMTRAL = RouteConfig(
    name = 'Las vías',
    arrival_rate_per_min = 1.0 / 3.0,
    peak_multiplier = 3.5,
    trip_duration_mean_min = 27.5,
    intermediate_stops = 3
)

ROUTES = [ROUTE_PUENTE_TUZOS, ROUTE_CEMTRAL]

SIMULATION_DAYS = 21
INTERVAL_MINUTES = 15
START_HOUR = 6
END_HOUR = 20
BUS_CAPACITY = 18
RANDOM_SEED = 42
PEAK_HOURS = [(6, 8), (11, 13)]
STOP_DELAY_SECONDS = 10.0  # 10 segundos extra por parada intermedia
AFTERNOON_MAX_WAIT_MINUTES = 15  # Después de las 12:00 PM, tiempo máximo de espera de 10-15 min