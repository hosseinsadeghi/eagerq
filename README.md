# eagerq
A simple vector-state gate-model simulator built as a personal hobby.

## Installation
Simply install the required packages and start using the repo by importing functions from the `eagerq` folder.

`pip install -r requirements.txt`

## Usage

Here is a simple example of how to create a GHZ state.

```python
from eagerq import Register

reg = Register(3)  # A register with three qubits
reg.h(0)  # Apply a Hadamard gate on the first qubit
reg.cnot(0, 1)  # Apply a CNOT gate with control qubit 0, and target qubit 1
reg.cnot(1, 2)  # Apply a CNOT gate with control qubit 1, and target qubit 2

print(reg)  # Print  the register. It will print out the current state.
```
```
0.7071|000⟩+0.7071|111⟩
```

## Fun facts
The name `eagerq` comes from the term "eager quantum" as in "eager execution". This means that when you apply a gate on a 
register, the gate takes effect immediately. Unfortunately, this means that further optimization on the application of 
gates cannot be done. Also, this library is not suitable for parameter tuning required for QAOA, VQE, and other QML algorithms.
