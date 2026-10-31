"""Simple live-demo UI for Quantum Checkpoint."""

from __future__ import annotations

import secrets

import matplotlib.pyplot as plt
import streamlit as st

from simulation import (
    available_checkpoints,
    available_targets,
    diagram_data,
    fault_trials,
    run_experiment,
    run_fault_trial,
    run_inside_t_comparison,
)


st.set_page_config(page_title="Quantum Checkpoint", page_icon="⚛", layout="wide")


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def pp(value):
    return "—" if value is None else f"{value:+.2f} pp"


def draw_circuit(option, fault_index=None):
    """Laptop-legible circuit sketch generated from the selected executed circuit."""
    circuit = option.protected_circuit
    width = max(11, min(16, len(circuit.data) * 0.55))
    figure, axis = plt.subplots(figsize=(width, 3.8))
    wire_labels = ("Data q0", "Data q1", "Helper q2")
    wire_count = circuit.num_qubits
    check_indices = set(option.check_operation_indices)
    moved_t_indices = set(option.moved_t_indices)

    for qubit, label in enumerate(wire_labels):
        y = wire_count - 1 - qubit
        axis.hlines(y, -0.6, len(circuit.data) + 1.8, color="#475569", linewidth=1.2)
        axis.text(-0.8, y, label, ha="right", va="center", fontsize=10, fontweight="bold")

    for index, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if not qubits:
            continue
        ys = [wire_count - 1 - qubit for qubit in qubits]
        is_check = index in check_indices
        is_moved_t = index in moved_t_indices
        fill = "#7c3aed" if is_check else ("#d97706" if is_moved_t else "#e2e8f0")
        edge = "#5b21b6" if is_check else ("#92400e" if is_moved_t else "#64748b")
        if len(ys) == 2:
            axis.vlines(index, min(ys), max(ys), color=edge, linewidth=1.7)
        for y in ys:
            axis.add_patch(plt.Rectangle((index - 0.31, y - 0.22), 0.62, 0.44, facecolor=fill, edgecolor=edge, linewidth=1))
            axis.text(index, y, instruction.operation.name.upper(), ha="center", va="center", fontsize=7,
                      color="white" if (is_check or is_moved_t) else "#172033")
        if is_check:
            axis.text(index, max(ys) + 0.36, f"L{option.operation_layers[index]}", ha="center", va="bottom",
                      fontsize=8, color="#5b21b6", fontweight="bold")
        if is_moved_t:
            axis.text(index, max(ys) + 0.36, "moved T", ha="center", va="bottom", fontsize=8,
                      color="#92400e", fontweight="bold")

    start, end = option.check_operation_indices[0], option.check_operation_indices[-1]
    axis.axvspan(start - 0.45, end + 0.45, color="#ede9fe", alpha=0.65, zorder=-2)
    top = wire_count - 0.25
    axis.annotate("Check starts", xy=(start, top - 0.15), xytext=(start - 1.0, top + 0.28), fontsize=9,
                  color="#5b21b6", arrowprops={"arrowstyle": "->", "color": "#5b21b6"})
    axis.annotate("Check ends", xy=(end, top - 0.15), xytext=(end + 0.25, top + 0.28), fontsize=9,
                  color="#5b21b6", arrowprops={"arrowstyle": "->", "color": "#5b21b6"})

    measure_x = len(circuit.data) + 0.75
    helper_y = wire_count - 1 - 2
    axis.add_patch(plt.Rectangle((measure_x - 0.25, helper_y - 0.24), 0.50, 0.48, facecolor="#0f766e", edgecolor="#115e59"))
    axis.text(measure_x, helper_y, "M", color="white", ha="center", va="center", fontweight="bold")
    axis.text(measure_x, helper_y - 0.48, "helper\nmeasurement", ha="center", va="top", fontsize=8, color="#0f766e")

    if fault_index is not None:
        fault_x = fault_index - 0.45
        target_y = wire_count - 1 - option.target
        axis.add_patch(plt.Rectangle((fault_x - 0.22, target_y - 0.20), 0.44, 0.40, facecolor="#dc2626", edgecolor="#991b1b"))
        axis.text(fault_x, target_y, "X", color="white", ha="center", va="center", fontweight="bold")
        axis.annotate("chosen X fault", xy=(fault_x, target_y + 0.2), xytext=(fault_x - 0.8, top + 0.22), fontsize=9,
                      color="#b91c1c", arrowprops={"arrowstyle": "->", "color": "#b91c1c"})

    axis.set_xlim(-1.4, len(circuit.data) + 1.6)
    axis.set_ylim(-0.8, wire_count + 0.65)
    axis.axis("off")
    figure.tight_layout()
    return figure


def add_history(records):
    st.session_state.history.extend(records)


def fresh_seed():
    st.session_state.sample_seed = secrets.randbelow(2_000_000_000) + 1


if "history" not in st.session_state:
    st.session_state.history = []
if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 20261031
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "fault_result" not in st.session_state:
    st.session_state.fault_result = None

st.title("⚛ Quantum Checkpoint")
st.caption("Choose one helper checkpoint. Keep the shots where helper q2 says the circuit looks clean.")

controls, diagram = st.columns([1, 1.55])
with controls:
    level = st.radio(
        "Circuit level",
        [1, 2],
        horizontal=True,
        format_func=lambda value: "Level 1 — Clifford" if value == 1 else "Level 2 — T gates",
    )
    target = st.radio("Data qubit", available_targets(level), horizontal=True, format_func=lambda q: f"Data q{q}")
    options = available_checkpoints(level, target)
    option_by_key = {option.key: option for option in options}
    checkpoint_key = st.selectbox(
        "Checkpoint interval",
        list(option_by_key),
        format_func=lambda key: option_by_key[key].title,
    )
    option = option_by_key[checkpoint_key]
    st.info(
        f"Helper **q2** checks **Data q{target}** from layer **{option.start_layer}** to **{option.end_layer}** "
        f"(spacing: **{option.separation} layers**)."
    )
    st.caption("For q0, Short and Long share the same start layer so only the spacing changes.")
    shots = st.selectbox("Shots", [100, 500, 1000, 2000], index=2, format_func=lambda value: f"{value:,}")
    seed = st.number_input("Sample seed", min_value=1, max_value=2_147_483_647, key="sample_seed", step=1)
    st.button("Fresh sample", width="stretch", on_click=fresh_seed)

with diagram:
    st.pyplot(draw_circuit(option), clear_figure=True)
    st.caption(
        f"Purple gates are actual Paulice helper-controlled Pauli interactions. **Syndrome q2 = 0** means keep the shot; "
        f"**q2 = 1** means flag it. Orange, when present, marks the moved Level-2 T gate. This executed circuit starts the check at L{option.start_layer}, ends it at L{option.end_layer}, then measures q2."
    )

if st.button(f"Run {shots:,} shots", type="primary", width="stretch"):
    with st.spinner("Running local Aer shots and exact density-matrix scores…"):
        latest = run_experiment(level, target, checkpoint_key, shots, int(seed))
    add_history([latest])

if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("Latest noisy experiment")
    sampled, predicted, fidelity = st.columns(3)
    with sampled:
        st.metric("Shots kept", f"{latest['kept']} / {latest['shots']}")
        st.metric("Shots flagged", latest["flagged"])
        st.metric("Sampled acceptance", percent(latest["sample_acceptance"]))
    with predicted:
        st.metric("Predicted acceptance", percent(latest["predicted_acceptance"]))
        st.caption("Prediction includes the same q2 readout error and syndrome-bit mapping used by the shots.")
    with fidelity:
        st.metric("Exact no-check data-state fidelity", percent(latest["no_check_fidelity"]))
        st.metric("Exact filtered data-state fidelity", percent(latest["filtered_fidelity"]), delta=pp(latest["fidelity_change_pp"]))
    if latest["filtered_fidelity"] is None:
        st.write("No reported-clean helper outcomes exist, so filtered data-state fidelity is unavailable.")
    else:
        direction = "improves" if latest["fidelity_change_pp"] > 0 else "does not improve"
        st.write(
            f"With {latest['checkpoint_key']}, filtering {direction} data-state fidelity by "
            f"{pp(latest['fidelity_change_pp'])} while the sample kept {percent(latest['sample_acceptance'])} of shots."
        )

st.subheader("Compare placements")
compare_a, compare_b, compare_c = st.columns(3)
with compare_a:
    if st.button("Compare short vs long", width="stretch"):
        with st.spinner("Running both intervals with the same seed and settings…"):
            comparison = [run_experiment(level, target, item.key, shots, int(seed)) for item in options]
        add_history(comparison)
        st.session_state.comparison = ("placements", comparison)
with compare_b:
    if st.button("Compare Clifford vs T", width="stretch"):
        with st.spinner("Running matched endpoints for both regular reference levels…"):
            comparison = [run_experiment(item_level, target, checkpoint_key, shots, int(seed)) for item_level in (1, 2)]
        add_history(comparison)
        st.session_state.comparison = ("levels", comparison)
with compare_c:
    disabled = target != 0
    if st.button("Try verified inside-T pair", disabled=disabled, width="stretch"):
        with st.spinner("Running the matched q0 S/T pair with one phase gate inside the check…"):
            st.session_state.comparison = ("inside_t", list(run_inside_t_comparison(checkpoint_key, shots, int(seed))))

if target != 0:
    st.caption("The optional matched inside-T pair is q0-only. Regular Level 2 now has one verified T(q1) before CX(q0, q1), plus five trailing T gates.")

if st.session_state.comparison:
    kind, values = st.session_state.comparison
    if kind == "placements":
        short, long = values
        st.write(
            f"At the same Level {level}, q{target}, seed {seed}: **{short['interval']}** kept {percent(short['sample_acceptance'])} "
            f"with {percent(short['filtered_fidelity'])} filtered fidelity, while **{long['interval']}** kept "
            f"{percent(long['sample_acceptance'])} with {percent(long['filtered_fidelity'])}."
        )
    elif kind == "levels":
        clifford, t_level = values
        st.write(
            f"At the same {checkpoint_key} endpoints, the Clifford reference has {percent(clifford['filtered_fidelity'])} filtered fidelity "
            f"and the T reference has {percent(t_level['filtered_fidelity'])}; their sampled acceptances are "
            f"{percent(clifford['sample_acceptance'])} and {percent(t_level['sample_acceptance'])}."
        )
    else:
        clifford, t_level = values
        st.write(
            f"Verified inside-check pair: with the same q0 endpoints, gate count, and noise, S gives "
            f"{percent(clifford['filtered_fidelity'])} and T gives {percent(t_level['filtered_fidelity'])} filtered data-state fidelity."
        )

with st.expander("Game mode — Will the checkpoint catch it?", expanded=False):
    fault_option, fault_options = fault_trials()
    trial_by_name = {trial.name: trial for trial in fault_options}
    trial_name = st.radio("Choose an X-fault location on Data q0", list(trial_by_name), horizontal=True)
    trial = trial_by_name[trial_name]
    st.pyplot(draw_circuit(fault_option, trial.insert_before), clear_figure=True)
    prediction = st.radio("Your prediction", ["Caught", "Missed"], horizontal=True)
    if st.button("Reveal helper result", type="secondary"):
        st.session_state.fault_result = (trial_name, prediction, run_fault_trial(trial_name))
    if st.session_state.fault_result and st.session_state.fault_result[0] == trial_name:
        _, predicted, result = st.session_state.fault_result
        outcome = result["actual"]
        st.write(f"You predicted **{predicted}**. The helper reports **{result['helper_result']}**: **{outcome}**.")
        if outcome == "Caught":
            st.write("The X occurs inside this checkpoint’s protected interval, so helper q2 flags it.")
        else:
            st.write(
                f"The X occurs after the check ends, so q2 still passes it even though the data-state fidelity falls to "
                f"{percent(result['data_fidelity_after_fault'])}."
            )
        st.caption("Background noise is off in this game mode, and these trials are not added to experiment history.")

if st.session_state.history:
    st.subheader("Experiment history")
    st.dataframe(
        [
            {
                "Level": f"Level {item['level']}", "Data": item["data_qubit"], "Start": item["start_layer"],
                "End": item["end_layer"], "Spacing": item["separation"], "Shots": item["shots"], "Seed": item["seed"],
                "Kept": item["kept"], "Flagged": item["flagged"], "Acceptance": percent(item["sample_acceptance"]),
                "Predicted": percent(item["predicted_acceptance"]), "No-check fidelity": percent(item["no_check_fidelity"]),
                "Filtered fidelity": percent(item["filtered_fidelity"]), "Change": pp(item["fidelity_change_pp"]),
            }
            for item in st.session_state.history
        ],
        width="stretch",
        hide_index=True,
    )

with st.expander("What is measured?"):
    st.markdown(
        "The two data qubits are **unencoded**, so scores are called **data-state fidelity**. "
        "Regular Level 2 moves one existing T(q1) before CX(q0, q1), then keeps five T gates after the check; every placement was verified with noise off. "
        "The q0-only inside-T comparison is separately verified: a matched S/T gate is inserted where the propagated Paulice check is Z, so T commutes with the check. "
        "Exact values come from a local Aer density matrix; sampled acceptance comes from the selected number of individual noisy shots."
    )
