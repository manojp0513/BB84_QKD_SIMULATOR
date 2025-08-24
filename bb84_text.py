import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# -------------------------------
# BB84 Protocol Implementation
# -------------------------------
def bb84_protocol(n_bits=32, eve_present=False):
    rng = np.random.default_rng()

    # Alice generates bits and bases
    alice_bits = rng.integers(2, size=n_bits)
    alice_bases = rng.integers(2, size=n_bits)

    # Prepare qubits
    circuits = []
    for bit, base in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1, 1)
        if base == 1:
            qc.h(0)
        if bit == 1:
            qc.x(0)
        circuits.append(qc)

    # Eve interception
    if eve_present:
        eve_bases = rng.integers(2, size=n_bits)
        for qc, eve_base in zip(circuits, eve_bases):
            if eve_base == 1:
                qc.h(0)
            qc.measure(0, 0)
            qc.reset(0)
            if eve_base == 1:
                qc.h(0)

    # Bob measures
    bob_bases = rng.integers(2, size=n_bits)
    for qc, bob_base in zip(circuits, bob_bases):
        if bob_base == 1:
            qc.h(0)
        qc.measure(0, 0)

    # Simulate
    simulator = AerSimulator()
    bob_results = []
    for qc in circuits:
        result = simulator.run(qc, shots=1).result()
        counts = result.get_counts()
        bob_results.append(int(max(counts, key=counts.get)))

    # Sifting
    sifted_bits_alice = []
    sifted_bits_bob = []
    for a_bit, a_base, b_base, b_bit in zip(alice_bits, alice_bases, bob_bases, bob_results):
        if a_base == b_base:
            sifted_bits_alice.append(a_bit)
            sifted_bits_bob.append(b_bit)

    # Error estimation
    revealed_count = max(1, len(sifted_bits_alice) // 4)
    revealed_indices = rng.choice(len(sifted_bits_alice), revealed_count, replace=False)
    revealed_errors = sum(
        sifted_bits_alice[i] != sifted_bits_bob[i] for i in revealed_indices
    )
    qber = revealed_errors / revealed_count if revealed_count > 0 else 0

    # Final key
    final_alice = [bit for i, bit in enumerate(sifted_bits_alice) if i not in revealed_indices]
    final_bob = [bit for i, bit in enumerate(sifted_bits_bob) if i not in revealed_indices]

    summary = {
        "n_bits_sent": n_bits,
        "n_sifted": len(sifted_bits_alice),
        "revealed_count": revealed_count,
        "qber": qber,
        "final_key_length": len(final_alice),
        "final_key_alice": "".join(map(str, final_alice)),
        "final_key_bob": "".join(map(str, final_bob))
    }

    # Combined big circuit
    big_circuit = QuantumCircuit(n_bits, n_bits)
    for i, (bit, base) in enumerate(zip(alice_bits, alice_bases)):
        if base == 1:
            big_circuit.h(i)
        if bit == 1:
            big_circuit.x(i)
        if eve_present:
            eve_base = rng.integers(2)
            if eve_base == 1:
                big_circuit.h(i)
            big_circuit.measure(i, i)
            big_circuit.reset(i)
            if eve_base == 1:
                big_circuit.h(i)
        if bob_bases[i] == 1:
            big_circuit.h(i)
        big_circuit.measure(i, i)

    return big_circuit, summary

# -------------------------------
# GUI Display Function
# -------------------------------
def display_gui(no_eve_circuit, no_eve_summary, eve_circuit, eve_summary):
    root = tk.Tk()
    root.title("BB84 QKD Simulator")
    root.geometry("1200x800")

    # Monospaced font for circuits
    font = ("Courier", 10)

    # --- NO EVE ---
    tk.Label(root, text="=== NO EVE ===", font=("Helvetica", 14, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
    circuit_text1 = scrolledtext.ScrolledText(root, width=80, height=15, font=font)
    circuit_text1.grid(row=1, column=0, padx=10)
    circuit_text1.insert(tk.END, no_eve_circuit.draw(output="text"))
    circuit_text1.config(state=tk.DISABLED)

    summary_text1 = scrolledtext.ScrolledText(root, width=40, height=15, font=font)
    summary_text1.grid(row=1, column=1, padx=10)
    qber_no = f"{no_eve_summary['qber']*100:.2f}%"
    if no_eve_summary['qber'] < 0.11:
        qber_no += " ✅ Secure"
    summary1 = (
        f"Bits sent: {no_eve_summary['n_bits_sent']}\n"
        f"Sifted bits: {no_eve_summary['n_sifted']}\n"
        f"Revealed bits: {no_eve_summary['revealed_count']}\n"
        f"QBER: {qber_no}\n"
        f"Final key length: {no_eve_summary['final_key_length']}\n"
        f"Alice key: {no_eve_summary['final_key_alice']}\n"
        f"Bob key:   {no_eve_summary['final_key_bob']}"
    )
    summary_text1.insert(tk.END, summary1)
    summary_text1.config(state=tk.DISABLED)

    # --- WITH EVE ---
    tk.Label(root, text="=== WITH EVE ===", font=("Helvetica", 14, "bold")).grid(row=2, column=0, sticky="w", padx=10, pady=5)
    circuit_text2 = scrolledtext.ScrolledText(root, width=80, height=15, font=font)
    circuit_text2.grid(row=3, column=0, padx=10)
    circuit_text2.insert(tk.END, eve_circuit.draw(output="text"))
    circuit_text2.config(state=tk.DISABLED)

    summary_text2 = scrolledtext.ScrolledText(root, width=40, height=15, font=font)
    summary_text2.grid(row=3, column=1, padx=10)
    qber_eve = f"{eve_summary['qber']*100:.2f}%"
    if eve_summary['qber'] > 0.11:
        qber_eve += " 🚨 Eve Detected!"
    summary2 = (
        f"Bits sent: {eve_summary['n_bits_sent']}\n"
        f"Sifted bits: {eve_summary['n_sifted']}\n"
        f"Revealed bits: {eve_summary['revealed_count']}\n"
        f"QBER: {qber_eve}\n"
        f"Final key length: {eve_summary['final_key_length']}\n"
        f"Alice key: {eve_summary['final_key_alice']}\n"
        f"Bob key:   {eve_summary['final_key_bob']}"
    )
    summary_text2.insert(tk.END, summary2)
    summary_text2.config(state=tk.DISABLED)

    root.mainloop()

# -------------------------------
# Run the Protocol
# -------------------------------
if __name__ == "__main__":
    no_eve_circuit, no_eve_summary = bb84_protocol(n_bits=32, eve_present=False)
    eve_circuit, eve_summary = bb84_protocol(n_bits=32, eve_present=True)
    display_gui(no_eve_circuit, no_eve_summary, eve_circuit, eve_summary)
