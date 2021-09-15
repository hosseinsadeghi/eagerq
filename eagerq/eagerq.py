"""
Copyright 2021 Hossein Sadeghi Esfahani

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: hosseinsadeghiesfahani@gmail.com
"""

from scipy.linalg import norm
from copy import deepcopy
from typing import Iterable
from eagerq.matrices import *
import numpy as np


class Register:
    """A qubit register class with eager gates.

    This class handles creation of a state vector and real-time application of various gates.
    This is a simple tool created as a hobby and it is not meant
    to be very performant at larger qubit counts (24+).

    Example:
        Create the superposition of two state for a single qubit

        >>> qubit = Register(1)
        >>> qubit.h(0)

        Create the superposition of all 4 states for 2 qubits

        >>> qubits = Register(2)
        >>> qubits.h(0).h(1)

        Create the GHZ state for three qubits

        >>> qubits = Register(3)
        >>> qubits.h(0).cnot(0, 1).cnot(1, 2)

    """

    def __init__(self, n, initial=None):
        if initial is None:
            self.n = n
            self.psi = np.zeros((2,) * n)
            self.psi[(0,) * n] = 1
        else:
            self.n = np.ndim(initial)
            self.psi = initial

    def h(self, i):
        """Add a Hadamard gate on qubit i"""
        self.psi = np.tensordot(_hadamard_matrix, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def cnot(self, control, target):
        """Add a CNOT gate between qubits control and target"""
        self.psi = np.tensordot(_cnot_tensor, self.psi, ((2, 3), (control, target)))
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))
        return self

    def ccnot(self, c1, c2, t):
        """Add a CCNOT gate between two control gates c1 and c2 and target t"""
        self.psi = np.tensordot(_toffoli_tensor, self.psi, ((3, 4, 5), (c1, c2, t)))
        self.psi = np.moveaxis(self.psi, (0, 1, 2), (c1, c2, t))
        return self

    def GHZ(self):
        """Add a group of CNOT to create a GHZ state"""
        for i in range(self.n - 1):
            self.cnot(i, i + 1)
        return self

    def x(self, i):
        """Add a X gate"""
        self.psi = np.tensordot(_paulix, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def y(self, i):
        """Add a Y gate"""
        self.psi = np.tensordot(_pauliy, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def z(self, i):
        """Add a Z gate"""
        self.psi = np.tensordot(_pauliz, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def __repr__(self):
        """Print the current vector state using ket notation"""
        tmp = self.psi.flatten()
        tmp = [(state, v) for state, v in enumerate(tmp) if v]
        return ''.join([f'{sign(i, v)}{np.round(v, 4)}|{convert(state, self.n)}\u27E9'
                        for i, (state, v) in enumerate(tmp)])

    def rx(self, i, theta):
        """Add a rotation gate RX(\theta)"""
        _rx = np.array(
            [[np.cos(theta / 2), - 1j * np.sin(theta / 2)],
             [- 1j * np.sin(theta / 2), np.cos(theta / 2)]])
        self.psi = np.tensordot(_rx, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def ry(self, i, theta):
        """Add a rotation gate RY(\theta)"""
        _ry = np.array(
            [[np.cos(theta / 2), - np.sin(theta / 2)],
             [np.sin(theta / 2), np.cos(theta / 2)]])
        self.psi = np.tensordot(_ry, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def rz(self, i, theta):
        """Add a rotation gate RZ(\theta)"""
        _rz = np.array(
            [[np.exp(- 1j * theta / 2), 0],
             [0, np.exp(1j * theta / 2)]])
        self.psi = np.tensordot(_rz, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)
        return self

    def xx(self, control, target, phi):
        """Add a XX gate between control and target with angle
            \phi, RXX(\phi)
        """
        self.psi = np.tensordot(_xx_tensor(phi), self.psi, ((2, 3), (control, target)))
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))
        return self


def convert(n, m):
    return f"{n:0{m}b}"


def sign(state, v):
    if state > 0 and v > 0:
        return '+'
    return ''
