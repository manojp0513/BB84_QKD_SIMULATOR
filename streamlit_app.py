# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bb84 import run_bb84

st.set_page_config(page_title="BB84 QKD Simulator", layout="wide")
st.title("BB84 Quantum Key Distribution — Simulator (Toy)")

st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider("Number of photons (N)", min_value=64, max_value=10000, value=1024, step=64)
p_noise = st.sidebar.slider("Channel noise (prob)", 0.0, 0.5, 0.01, 0.01)
eve_on = st.sidebar.checkbox("Eve (intercept-resend) ON", value=False)
p_eve = st.sidebar.slider("Eve intercept probability", 0.0, 1.0, 0.2, 0.05)
sample_frac = st.sidebar.slider("Sample fraction for QBER", 0.0, 0.5, 0.1, 0.01)
qber_threshold = st.sidebar.slider("QBER abort threshold", 0.0, 0.5, 0.11, 0.01)
block_size = st.sidebar.number_input("Reconciliation block size", value=16, min_value=4, max_value=512, step=4)
final_bits = st.sidebar.number_input("Final key length (bits)", value=128, min_value=32, max_value=256, step=8)
seed = st.sidebar.number_input("Random seed (0 = random)", value=0, step=1)
seed_val = None if seed == 0 else int(seed)

run_button = st.sidebar.button("Run Simulation")

st.markdown("## Controls")
st.markdown("Use the sidebar to set parameters. Toggle Eve on to see QBER jump and protocol abort (or produce fewer secure bits).")

if run_button:
    with st.spinner("Running BB84 simulation..."):
        metrics, outputs ,log = run_bb84(
            N=N,
            p_noise=p_noise,
            eve_on=eve_on,
            p_eve=p_eve if eve_on else 0.0,
            sample_frac=sample_frac,
            qber_threshold=qber_threshold,
            reconcile_block_size=block_size,
            final_key_bits=final_bits,
            seed=seed_val
        )

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Simulation Metrics")
        df = pd.DataFrame(list(metrics.items()), columns=["Metric","Value"])
        st.table(df)

        st.subheader("Key / Status")
        if metrics.get("aborted"):
            st.error(f"Protocol aborted: {metrics.get('reason')}")
        elif outputs.get("final_key"):
            st.success("Protocol finished successfully.")
            st.code(outputs["final_key"], language="text")
        else:
            st.warning("No final key (not enough sifted bits).")

    with col2:
        st.subheader("Quick Charts")
        keys = ["N", "same_basis_count", "sifted_key_length_after_reveal", "final_key_bits"]
        values = [metrics.get(k, 0) for k in keys]
        fig, ax = plt.subplots(figsize=(3.5,2.5))
        ax.bar(keys, values)
        ax.set_ylabel("count")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    st.subheader("Explanation / Notes")
    st.markdown("""
    - **Same-basis count** is number of transmissions where Alice & Bob used the same basis (expected ~N/2).
    - **QBER** estimated from a random sample; high QBER indicates eavesdropping or noise.
    - **Reconciliation** is a toy parity-block method (illustrative only).
    - **Privacy amplification**: SHA-256 + truncation to produce final key.
    - This simulator is *educational* — not for production cryptography.
    """)

    st.markdown("### Want to re-run quickly?")
    if st.button("Rerun same settings"):
        st.experimental_rerun()
else:
    st.info("Press 'Run Simulation' in the sidebar when you're ready.")
