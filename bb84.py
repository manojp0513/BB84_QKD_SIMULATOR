# bb84.py
# Core BB84 simulator functions (educational toy simulator)
import numpy as np
import hashlib
import json
from typing import Tuple, Dict, Any, List

def measure_state(state, measure_basis, rng):
    sent_basis, bit = state
    if int(measure_basis) == int(sent_basis):
        return int(bit)
    else:
        # measurement in wrong basis -> random bit
        return int(rng.integers(0,2))

def flip_state(state):
    basis, bit = state
    return (basis, 1-bit)

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def parity_block_reconcile(alice: np.ndarray, bob: np.ndarray, block_size: int = 16):
    """
    Toy parity-based reconciliation to demonstrate error-correction messaging.
    Not secure/production-grade.
    """
    n = len(alice)
    corrected = bob.copy()
    comm_cost = 0
    for i in range(0, n, block_size):
        a_block = alice[i:i+block_size]
        b_block = corrected[i:i+block_size]
        a_par = int(np.sum(a_block) % 2)
        b_par = int(np.sum(b_block) % 2)
        comm_cost += 1
        if a_par != b_par:
            # binary-search style locate single-bit error (toy)
            lo = i
            hi = min(n, i+block_size) - 1
            while lo <= hi:
                if lo == hi:
                    corrected[lo] = 1 - corrected[lo]
                    break
                mid = (lo + hi) // 2
                a_left_par = int(np.sum(alice[lo:mid+1]) % 2)
                b_left_par = int(np.sum(corrected[lo:mid+1]) % 2)
                comm_cost += 1
                if a_left_par != b_left_par:
                    hi = mid
                else:
                    lo = mid + 1
    return corrected, comm_cost

def privacy_amplification(key_bits: np.ndarray, output_len_bits: int = 128) -> str:
    """
    Hash and truncate to produce final key hex string.
    """
    bits = ''.join(str(int(b)) for b in key_bits.tolist())
    pad = (8 - (len(bits) % 8)) % 8
    bits_padded = bits + '0' * pad
    b = int(bits_padded, 2).to_bytes(len(bits_padded)//8, byteorder='big')
    digest = hashlib.sha256(b).digest()
    bitstring = ''.join(f'{byte:08b}' for byte in digest)
    truncated = bitstring[:output_len_bits]
    hexlen = (len(truncated)+3)//4
    return hex(int(truncated,2))[2:].zfill(hexlen)

def run_bb84(
    N: int = 1024,
    p_noise: float = 0.0,
    eve_on: bool = False,
    p_eve: float = 0.0,
    sample_frac: float = 0.1,
    qber_threshold: float = 0.11,
    reconcile_block_size: int = 16,
    final_key_bits: int = 128,
    seed: int = None
) -> Tuple[Dict[str,Any], Dict[str,Any], List[Dict[str,Any]]]:
    rng = np.random.default_rng(seed)

    log = []  # 👈 JSON-friendly log of steps

    # Alice
    alice_bits = rng.integers(0,2,size=N)
    alice_bases = rng.integers(0,2,size=N)  # 0='+', 1='x'
    sent_states = list(zip(alice_bases.tolist(), alice_bits.tolist()))
    log.append({"step": "Alice generates bits", "value": alice_bits.tolist()[:20]})
    log.append({"step": "Alice chooses bases", "value": alice_bases.tolist()[:20]})

    # Eve intercept-resend simulation
    eve_bases = np.full(N, -1, dtype=int)
    if eve_on and p_eve > 0:
        for i in range(N):
            if rng.random() < p_eve:
                eb = int(rng.integers(0,2))
                eve_bases[i] = eb
                eve_bit = measure_state(sent_states[i], eb, rng)
                sent_states[i] = (eb, eve_bit)
        log.append({"step": "Eve intercepts", "value": f"p_eve={p_eve}"})

    # Channel noise (bit flips)
    flips = 0
    for i in range(N):
        if rng.random() < p_noise:
            sent_states[i] = flip_state(sent_states[i])
            flips += 1
    log.append({"step": "Noise applied", "value": f"{flips} flips"})

    # Bob measures
    bob_bases = rng.integers(0,2,size=N)
    bob_bits = np.zeros(N, dtype=int)
    for i in range(N):
        bob_bits[i] = measure_state(sent_states[i], int(bob_bases[i]), rng)
    log.append({"step": "Bob measures", "value": bob_bits.tolist()[:20]})

    # Sifting
    same_basis_mask = (alice_bases == bob_bases)
    indices_kept = np.nonzero(same_basis_mask)[0]
    alice_sift = alice_bits[same_basis_mask]
    bob_sift = bob_bits[same_basis_mask]
    n_sift_total = len(alice_sift)
    log.append({"step": "Sifting", "value": f"{n_sift_total} bits kept"})

    # QBER estimation sample
    sample_size = max(0, int(np.floor(sample_frac * n_sift_total)))
    sample_indices = rng.choice(n_sift_total, size=sample_size, replace=False) if sample_size>0 else np.array([],dtype=int)
    if sample_size > 0:
        sample_errors = hamming(alice_sift[sample_indices], bob_sift[sample_indices])
        qber = sample_errors / sample_size
    else:
        sample_errors = 0
        qber = 0.0
    log.append({"step": "QBER estimation", "value": f"{qber:.3f}"})

    # Remove revealed sample bits
    if sample_size > 0:
        mask = np.ones(n_sift_total, dtype=bool)
        mask[sample_indices] = False
        alice_sift = alice_sift[mask]
        bob_sift = bob_sift[mask]

    aborted = False
    reason = ""
    if qber > qber_threshold:
        aborted = True
        reason = f"QBER too high ({qber:.3f} > {qber_threshold}) — possible eavesdropping"
        metrics = {
            "N": N,
            "same_basis_count": int(n_sift_total),
            "sifted_key_length_after_reveal": int(len(alice_sift)),
            "sample_size": sample_size,
            "sample_errors": int(sample_errors),
            "qber": float(qber),
            "aborted": True,
            "reason": reason,
            "eve_on": eve_on,
            "p_eve": p_eve,
            "p_noise": p_noise
        }
        outputs = {"final_key": None}
        return metrics, outputs, log

    # Reconciliation (toy)
    if len(alice_sift) == 0:
        metrics = {
            "N": N,
            "same_basis_count": int(n_sift_total),
            "sifted_key_length_after_reveal": 0,
            "sample_size": sample_size,
            "sample_errors": int(sample_errors),
            "qber": float(qber),
            "aborted": False,
            "reason": "No sifted bits left",
            "comm_cost": 0,
            "eve_on": eve_on,
            "p_eve": p_eve,
            "p_noise": p_noise
        }
        outputs = {"final_key": None}
        return metrics, outputs, log

    bob_corrected, comm_cost = parity_block_reconcile(alice_sift, bob_sift, block_size=reconcile_block_size)
    log.append({"step": "Reconciliation", "value": f"{comm_cost} messages"})

    final_key_hex = privacy_amplification(bob_corrected, output_len_bits=final_key_bits)
    log.append({"step": "Final key", "value": final_key_hex})

    metrics = {
        "N": N,
        "same_basis_count": int(n_sift_total),
        "sifted_key_length_after_reveal": int(len(alice_sift)),
        "sample_size": sample_size,
        "sample_errors": int(sample_errors),
        "qber": float(qber),
        "aborted": False,
        "reason": reason,
        "comm_cost": comm_cost,
        "final_key_bits": final_key_bits,
        "final_key_hex_len": len(final_key_hex),
        "eve_on": eve_on,
        "p_eve": p_eve,
        "p_noise": p_noise
    }

    outputs = {
        "final_key": final_key_hex,
        "alice_sift": alice_sift.tolist(),
        "bob_sift_before_reconcile": bob_sift.tolist(),
        "bob_after_reconcile": bob_corrected.tolist(),
        "indices_kept": indices_kept.tolist()
    }

    return metrics, outputs, log


# quick test run when module executed directly
if __name__ == "__main__":
    m,o,log = run_bb84(N=128, p_noise=0.01, eve_on=True, p_eve=0.2, seed=42)
    print("METRICS:", json.dumps(m, indent=2))
    print("FINAL KEY:", o["final_key"])
    with open("result.json","w") as f:
        json.dump(log, f, indent=2)   # 👈 full simulation step log saved
    print("Step-by-step log written to result.json")
