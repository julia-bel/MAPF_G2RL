import numpy as np


def manhattan_distance(x_st: int, y_st: int, x_end: int, y_end: int) -> int:
    return abs(x_end - x_st) + abs(y_end - y_st)
    
    
def moving_cost(num_steps: int, c_start: list[int], c_goal: list[int]) -> float:
    return num_steps / (manhattan_distance(*c_start, *c_goal))


def detour_percentage(num_steps: int, opt_path_len: int) -> float:
    return (num_steps - opt_path_len) / opt_path_len * 100
