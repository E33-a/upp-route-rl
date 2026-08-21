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
# Flota total: 35 a 45 unidades (80-90% operando en hora pico)
# Volumen total de demanda: > 4,500 alumnos diarios
ROUTES = [
    RouteConfig(
        name="Las vías",
        arrival_rate_per_min=1.2,           # ~4.2 pax/min en hora pico (se llena en 3-5 min)
        trip_duration_mean_min=28.0,
        peak_multiplier=3.5,
        intermediate_stops=3,
        base_start_hour=6.0,
        base_end_hour=13.0,          # Hacen base hasta las 13:00 hrs
        last_departure_hour=18.5       # Última combi sale aprox 18:30 hrs
    ),
    RouteConfig(
        name="Puente Tuzos",
        arrival_rate_per_min=0.5,           # ~1.25 pax/min en hora pico
        trip_duration_mean_min=22.0,
        peak_multiplier=2.5,
        intermediate_stops=2,
        base_start_hour=6.0,
        base_end_hour=12.0,          # Hacen base hasta las 12:00 hrs
        last_departure_hour=14.0       # Pasan en tránsito en la tarde
    )
]

SIMULATION_DAYS = 21
INTERVAL_MINUTES = 15
START_HOUR = 6
END_HOUR = 20
BUS_CAPACITY = 18
RANDOM_SEED = 42
PEAK_HOURS = [(6, 9), (11, 13)]  # Hora pico matutina real: 06:00 a 09:00 hrs
STOP_DELAY_SECONDS = 10.0  # 10 segundos extra por parada intermedia
AFTERNOON_MAX_WAIT_MINUTES = 30.0  # Tiempo máximo de espera de 30 minutos en hora valle