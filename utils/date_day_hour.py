import numpy as np
def encode_day_cyclic(day_num: int) -> tuple[float, float]:
    sin_day = np.sin(2 * np.pi * day_num / 7)
    cos_day = np.cos(2 * np.pi * day_num / 7)
    return sin_day, cos_day

