import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class UCB1Agent:
    """
    Agente Algoritmo UCB1 (Upper Confidence Bound 1) para la optimización
    dinámica de políticas de despacho de combis en la UPP.
    """
    
    def __init__(self, n_arms: int = 5, c: float = 1.414):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)     # N(a): número de veces que se eligió la acción a
        self.rewards = np.zeros(n_arms, dtype=float)   # Suma acumulada de recompensas para la acción a
        self.means = np.zeros(n_arms, dtype=float)     # Q(a): recompensa promedio estimada
        self.total_steps = 0
        
        # Nombres descriptivos de las 5 políticas (brazos) requeridas por la profesora
        self.arm_names = [
            "a0: Tradicional (18 pax rígidamente)",
            "a1: Flex (>=14 pax tras 15 min)",
            "a2: Flex (>=12 pax tras 25 min)",
            "a3: Flex (>=10 pax tras 35 min)",
            "a4: Salida Máxima (<10 pax tras 45 min)"
        ]

    def select_action(self) -> int:
        """
        Selecciona la mejor acción a mediante la fórmula UCB1:
        Q_ucb(a) = Q_bar(a) + c * sqrt(ln(t) / N(a))
        """
        self.total_steps += 1
        
        # Fase de exploración inicial: probar cada brazo al menos una vez
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        
        # Calcular el término UCB1 para todos los brazos
        ucb_values = self.means + self.c * np.sqrt(np.log(self.total_steps) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        """
        Actualiza el conteo y la recompensa promedio acumulada para el brazo seleccionado.
        """
        self.counts[action] += 1
        self.rewards[action] += reward
        self.means[action] = self.rewards[action] / self.counts[action]


def evaluate_policy_dispatch(action: int, students_waiting: int, wait_time_min: float) -> Tuple[bool, int, int]:
    """
    Evalúa si la regla de la política (brazo) se cumple dado el estado actual del paradero.
    
    Retorna: (se_despacha: bool, pasajeros_abordados: int, asientos_vacios: int)
    """
    bus_capacity = 18
    should_dispatch = False
    
    if action == 0:
        # a0: Salir solo con combi llena (18 pax)
        should_dispatch = (students_waiting >= bus_capacity)
    elif action == 1:
        # a1: >= 14 personas y pasaron 15 min (o 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 14 and wait_time_min >= 15.0)
    elif action == 2:
        # a2: >= 12 personas y pasaron 25 min (o 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 12 and wait_time_min >= 25.0)
    elif action == 3:
        # a3: >= 10 personas y pasaron 35 min (o 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 10 and wait_time_min >= 35.0)
    elif action == 4:
        # a4: < 10 personas y pasaron 45 min (o 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 6 and wait_time_min >= 45.0)

    if should_dispatch and students_waiting > 0:
        passengers = min(students_waiting, bus_capacity)
        empty_seats = bus_capacity - passengers
        return True, passengers, empty_seats
    
    return False, 0, bus_capacity


def calculate_reward(
    wait_time_min: float,
    leftover_students: int,
    empty_seats: int,
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.2,
    max_wait_norm: float = 60.0,
    max_leftover_norm: float = 30.0,
    max_empty_norm: float = 18.0
) -> float:
    """
    Calcula la recompensa R en el rango [0, 1] basada en la función de costo:
    P = w1 * wait_time_norm + w2 * leftover_norm + w3 * empty_seats_norm
    R = 1.0 - P
    """
    # Normalización min-max de cada componente entre 0 y 1
    norm_wait = min(1.0, wait_time_min / max_wait_norm)
    norm_leftover = min(1.0, leftover_students / max_leftover_norm)
    norm_empty = min(1.0, empty_seats / max_empty_norm)
    
    penalty = (w1 * norm_wait) + (w2 * norm_leftover) + (w3 * norm_empty)
    reward = float(np.clip(1.0 - penalty, 0.0, 1.0))
    return reward
