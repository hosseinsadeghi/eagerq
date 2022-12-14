import numpy as np

__all__ = [
    '_xx_tensor', '_hadamard_matrix', '_cnot_tensor', '_cz_tensor',
    '_paulix', '_pauliz', '_pauliy', '_swap_tensor', '_toffoli_tensor',
    '_phase', '_pi8th'
]


def _xx_tensor(phi):
    cosphi = np.cos(phi / 2)
    sinphi = np.sin(phi / 2)
    return np.reshape(
        [[cosphi, 0, 0, - 1j * sinphi],
         [0, cosphi, - 1j * sinphi, 0],
         [0, - 1j * sinphi, cosphi, 0],
         [- 1j * sinphi, 0, 0, cosphi]],
        (2, 2, 2, 2)
    )


_hadamard_matrix = 1 / np.sqrt(2) * np.array([[1, 1],
                                              [1, -1]])
_cnot_matrix = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])
_cnot_tensor = np.reshape(_cnot_matrix, (2, 2, 2, 2))
_cz_tensor = np.reshape(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    (2, 2, 2, 2)
)

_swap_tensor = np.reshape(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    (2, 2, 2, 2)
)
_toffoli_tensor = np.eye(8)
_toffoli_tensor[-2:, -2:] = [[0, 1], [1, 0]]
_toffoli_tensor = np.reshape(_toffoli_tensor, (2, 2, 2, 2, 2, 2))

_pauliy = np.array([[0, -1j], [1j, 0]])
_pauliz = np.array([[1, 0], [0, -1]])
_paulix = np.array([[0, 1], [1, 0]])
_phase = np.array([[1, 0], [0, 1j]])
_pi8th = np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]])
