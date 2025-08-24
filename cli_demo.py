# cli_demo.py
import argparse
import pprint
from bb84 import run_bb84

def main():
    parser = argparse.ArgumentParser(description="BB84 Simulator CLI Demo")
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--p_noise", type=float, default=0.0)
    parser.add_argument("--eve_on", action="store_true")
    parser.add_argument("--p_eve", type=float, default=0.0)
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--qber_threshold", type=float, default=0.11)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--final_bits", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    metrics, outputs = run_bb84(
        N=args.N,
        p_noise=args.p_noise,
        eve_on=args.eve_on,
        p_eve=args.p_eve,
        sample_frac=args.sample_frac,
        qber_threshold=args.qber_threshold,
        reconcile_block_size=args.block_size,
        final_key_bits=args.final_bits,
        seed=args.seed
    )

    print("\n=== BB84 Simulation Results ===")
    pprint.pprint(metrics)
    if outputs.get("final_key"):
        print("\nFinal Key (hex):", outputs["final_key"])
    else:
        print("\nNo final key generated (aborted or no bits).")
    print("\n(You can re-run with different --p_eve/--p_noise to demo Eve/no-Eve.)\n")

if __name__ == "__main__":
    main()
