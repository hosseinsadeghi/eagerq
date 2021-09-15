from scipy.linalg import norm
from copy import deepcopy
import numpy as np


_projectors = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]


def project(i, j, reg):
    projected = np.tensordot(_projectors[j], reg.psi, (1, i))
    return np.moveaxis(projected, 0, i)


def measure(i, reg):
    projected = project(i, 0, reg)
    norm_projected = norm(projected.flatten())
    norm_value = norm_projected if norm_projected > 0 else 1

    if np.random.random() < norm_projected ** 2:
        reg.psi = projected / norm_value
        return 0
    else:
        projected = project(i, 1, reg)
        reg.psi = projected / norm_value
        return 1


def measure_all(reg, shots=1, copy_state=True):
    result = []
    for _ in range(shots):
        if copy_state:
            reg_to_measure = deepcopy(reg)
        else:
            reg_to_measure = reg
        state = []
        for i in range(reg.n):
            state.append(measure(i, reg_to_measure))
        result.append(state)
    return result
