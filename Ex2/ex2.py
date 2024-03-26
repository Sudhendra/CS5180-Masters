import numpy as np
import random
from typing import Tuple, Dict, List
from enum import IntEnum

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

WALLS = [
    (0, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 0),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 10),
    (6, 4),
    (7, 4),
    (9, 4),
    (10, 4),
]
GOAL_STATE = (10, 10)
START_STATE = (0, 0)

ACTION_DX_DY = {
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
}

def is_wall_or_boundary(state: Tuple[int, int]) -> bool:
    x, y = state
    return state in WALLS or x < 0 or y < 0 or x >= 11 or y >= 11

def generate_transition_table_corrected():
    transition_table = []

    for x in range(11):
        for y in range(11):
            current_state = (x, y)
            if current_state in WALLS:
                continue

            for action in Action:
                dx, dy = ACTION_DX_DY[action]
                intended_next_state = (x + dx, y + dy)
                if intended_next_state in WALLS or not (0 <= intended_next_state[0] < 11) or not (0 <= intended_next_state[1] < 11):
                    intended_next_state = current_state 
                transition_table.append({
                    'from': current_state,
                    'action': action,
                    'to': intended_next_state,
                    'reward': 0,
                    'probability': 0.8
                })

                # Handle slips
                slip_probabilities = [0.1, 0.1]  # Two slips at 0.1 probability each
                for slip_prob in slip_probabilities:
                    # Slip transitions result in the same state
                    transition_table.append({
                        'from': current_state,
                        'action': action,
                        'to': current_state,
                        'reward': 0,
                        'probability': slip_prob
                    })

    transition_table.append({
        'from': GOAL_STATE,
        'action': None,
        'to': START_STATE,
        'reward': 1,
        'probability': 1.0
    })

    return transition_table

transition_table = generate_transition_table_corrected()
print(f"Number of non-zero transitions: {len(transition_table)}")

import pandas as pd

table = pd.DataFrame(transition_table)
table.to_csv('transition_probabilities.csv', index=False)