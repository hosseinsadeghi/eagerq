import numpy as np
from eagerq import Register

def test_ghz_3_qubit():
    """Test that a 3-qubit GHZ state is correctly created."""
    reg = Register(3)
    reg.h(0)
    reg.cnot(0, 1)
    reg.cnot(1, 2)
    state = reg.psi.flatten()
    # GHZ state should be (|000> + |111>) / sqrt(2)
    expected = np.zeros(8)
    expected[0] = 1 / np.sqrt(2)  # |000>
    expected[7] = 1 / np.sqrt(2)  # |111>
    np.testing.assert_allclose(state, expected, atol=1e-10)

def test_ghz_method():
    """Test the built-in GHZ method."""
    reg = Register(3)
    reg.h(0)
    reg.GHZ()
    state = reg.psi.flatten()
    expected = np.zeros(8)
    expected[0] = 1 / np.sqrt(2)
    expected[7] = 1 / np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-10)
