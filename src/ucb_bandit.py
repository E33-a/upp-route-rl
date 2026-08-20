import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class UCB1Agent:
    """
    UCB1 (Upper Confidence Bound 1) Multi-Armed Bandit Agent for dynamic
    van dispatch policy optimization at Universidad Politécnica de Pachuca (UPP).
    """
    
    def __init__(self, n_arms: int = 5, c: float = 1.414):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)     # N(a): number of times action a was selected
        self.rewards = np.zeros(n_arms, dtype=float)   # Total cumulative reward for action a
        self.means = np.zeros(n_arms, dtype=float)     # Q(a): estimated mean reward for action a
        self.total_steps = 0
        
        # Descriptive titles for the 5 dispatch policy arms specified in the course rubric
        self.arm_names = [
            "a0: Traditional (18 pax strictly)",
            "a1: Flex (>=14 pax after 15 min)",
            "a2: Flex (>=12 pax after 25 min)",
            "a3: Flex (>=10 pax after 35 min)",
            "a4: Max Exit (<10 pax after 45 min)"
        ]

    def select_action(self) -> int:
        """
        Select action 'a' using the UCB1 decision formula:
        Q_ucb(a) = Q_bar(a) + c * sqrt(ln(t) / N(a))
        """
        self.total_steps += 1
        
        # Initial exploration phase: try each arm at least once
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        
        # Compute UCB values for all candidate arms
        ucb_values = self.means + self.c * np.sqrt(np.log(self.total_steps) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        """
        Update action counts and rolling mean reward for the selected arm.
        """
        self.counts[action] += 1
        self.rewards[action] += reward
        self.means[action] = self.rewards[action] / self.counts[action]


def evaluate_policy_dispatch(action: int, students_waiting: int, wait_time_min: float) -> Tuple[bool, int, int]:
    """
    Evaluates whether the policy rule (arm) triggers a vehicle dispatch given the current station state.
    
    Returns: (should_dispatch: bool, passengers_boarded: int, empty_seats: int)
    """
    bus_capacity = 18
    should_dispatch = False
    
    if action == 0:
        # a0: Dispatch only when vehicle is full (18 pax)
        should_dispatch = (students_waiting >= bus_capacity)
    elif action == 1:
        # a1: >= 14 passengers and wait time >= 15 min (or 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 14 and wait_time_min >= 15.0)
    elif action == 2:
        # a2: >= 12 passengers and wait time >= 25 min (or 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 12 and wait_time_min >= 25.0)
    elif action == 3:
        # a3: >= 10 passengers and wait time >= 35 min (or 18 pax)
        should_dispatch = (students_waiting >= bus_capacity) or (students_waiting >= 10 and wait_time_min >= 35.0)
    elif action == 4:
        # a4: < 10 passengers and wait time >= 45 min (or 18 pax)
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
    Calculates normalized reward R in [0, 1] based on weighted operational penalty:
    Penalty = w1 * wait_norm + w2 * leftover_norm + w3 * empty_seats_norm
    Reward = 1.0 - Penalty
    """
    # Min-max normalization for each operational component
    norm_wait = min(1.0, wait_time_min / max_wait_norm)
    norm_leftover = min(1.0, leftover_students / max_leftover_norm)
    norm_empty = min(1.0, empty_seats / max_empty_norm)
    
    penalty = (w1 * norm_wait) + (w2 * norm_leftover) + (w3 * norm_empty)
    reward = float(np.clip(1.0 - penalty, 0.0, 1.0))
    return reward
