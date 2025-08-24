from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, circuit_drawer
import matplotlib.pyplot as plt
import random

# -------------------------
# Step 1: Alice prepares qubits
# -------------------------
n = 4  # small demo with 4 qubits
alice_bits = [random.randint(0, 1) for _ in range(n)]
alice_bases = [random.randint(0, 1) for _ in range(n)]  # 0=Z basis, 1=X basis

circuits = []

for i in range(n):
    qc = QuantumCircuit(1, 1)

    # Prepare bit
    if alice_bits[i] == 1:
        qc.x(0)  # bit=1 → apply X gate

    # Choose basis
    if alice_bases[i] == 1:
        qc.h(0)  # switch to X basis using Hadamard

    circuits.append(qc)

# -------------------------
# Step 2: Bob measures with random bases
# -------------------------
bob_bases = [random.randint(0, 1) for _ in range(n)]
measurements = []

backend = Aer.get_backend("qasm_simulator")
all_counts = []

for i in range(n):
    qc = circuits[i].copy()

    if bob_bases[i] == 1:
        qc.h(0)  # measure in X basis

    qc.measure(0, 0)

    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=1000).result()  # run with 1000 shots for histogram
    counts = result.get_counts()
    all_counts.append(counts)

    measured_bit = int(max(counts, key=counts.get))  # take most common measurement
    measurements.append(measured_bit)

# -------------------------
# Step 3: Compare Bases
# -------------------------
sifted_key = []
for i in range(n):
    if alice_bases[i] == bob_bases[i]:
        sifted_key.append(measurements[i])

print("Alice bits:   ", alice_bits)
print("Alice bases:  ", alice_bases)
print("Bob bases:    ", bob_bases)
print("Bob results:  ", measurements)
print("Sifted key:   ", sifted_key)

# -------------------------
# Step 4: Save Circuit Diagram (PNG)
# -------------------------
example_circuit = circuits[0]
example_circuit.measure(0, 0)
circuit_drawer(example_circuit, output="mpl", filename="bb84_circuit.png")
print("✅ Circuit diagram saved as bb84_circuit.png")

# -------------------------
# Step 5: Save Histogram (PNG)
# -------------------------
plot_histogram(all_counts[0])
plt.savefig("bb84_histogram.png")
print("✅ Histogram saved as bb84_histogram.png")
