"""Verified local Qiskit Aer simulations for the Quantum Checkpoint game.

Paulice builds the Clifford spacetime checks. Every playable checkpoint uses q2
as its one helper and is verified noiselessly before it is exposed in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit_aer.noise import ReadoutError, depolarizing_error
from qiskit_paulice import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel


DATA_QUBITS = (0, 1)
HELPER_QUBIT = 2
PAULICE_SEED = 2026
READOUT_FLIP = 0.015
POST_PHASE_QUBITS = (0, 1, 0, 1, 0, 1)


@dataclass(frozen=True)
class Checkpoint:
    """One real, one-helper Paulice check in the circuit that will run."""

    level: int
    key: str
    interval: str
    target: int
    start_layer: int
    end_layer: int
    separation: int
    protected_circuit: QuantumCircuit
    base_check_circuit: QuantumCircuit
    moved_t_indices: tuple[int, ...]
    check_operation_indices: tuple[int, ...]
    operation_layers: tuple[int, ...]
    paulice_cost: float | None

    @property
    def title(self) -> str:
        return f"{self.interval.title()} — layers {self.start_layer} to {self.end_layer}"


@dataclass(frozen=True)
class FaultTrial:
    name: str
    description: str
    insert_before: int
    expected: str


def _payload() -> QuantumCircuit:
    """Fixed two-data-qubit Clifford payload used by both regular levels."""
    circuit = QuantumCircuit(2, name="fixed_payload")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(1)
    circuit.cx(1, 0)
    return circuit


def _append_post_phase_block(circuit: QuantumCircuit, level: int) -> None:
    """Original reference: six matched S/T gates after the check interval."""
    gate = circuit.s if level == 1 else circuit.t
    for qubit in POST_PHASE_QUBITS:
        gate(qubit)


def _insert_moved_level2_t(circuit: QuantumCircuit) -> QuantumCircuit:
    """Move one existing T(q1) to the fixed boundary immediately before CX(q0, q1).

    The q1-long Paulice circuit places this after its first q2/q1 interaction and
    before the later q2/q1 interactions.  The following CX acts on the phase,
    so this is a genuine non-Clifford placement rather than a trailing phase.
    """
    result = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    inserted = False
    for instruction in circuit.data:
        qargs = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if not inserted and instruction.operation.name == "cx" and qargs == [0, 1]:
            result.t(1)
            inserted = True
        result.append(instruction.operation, [result.qubits[index] for index in qargs], [])
    if not inserted:
        raise RuntimeError("Could not find the fixed payload CX(q0, q1) boundary for moved T(q1).")
    return result


def _regular_level_circuit(base_circuit: QuantumCircuit, level: int) -> QuantumCircuit:
    """Apply the fixed regular level payload to a Paulice circuit or bare payload."""
    if level == 1:
        result = base_circuit.copy()
        _append_post_phase_block(result, 1)
        return result

    # Move the second original trailing gate, T(q1), inside. The other five remain
    # at the original post-check phase block, so Level 2 still has exactly six T gates.
    result = _insert_moved_level2_t(base_circuit)
    for index, qubit in enumerate(POST_PHASE_QUBITS):
        if index != 1:
            result.t(qubit)
    return result


def _moved_t_indices(circuit: QuantumCircuit, level: int) -> tuple[int, ...]:
    """Locate the one regular Level-2 T that is before the payload CX(q0, q1)."""
    if level != 2:
        return ()
    for index, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if instruction.operation.name != "t" or qubits != [1]:
            continue
        for later in circuit.data[index + 1 :]:
            later_qubits = [circuit.find_bit(qubit).index for qubit in later.qubits]
            if later.operation.name == "cx" and later_qubits == [0, 1]:
                return (index,)
    raise RuntimeError("Could not locate the regular Level-2 moved T(q1).")


def _regular_ideal_state(level: int) -> np.ndarray:
    circuit = _regular_level_circuit(_payload(), level)
    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)


def _paulice_search_noise() -> PauliceNoiseModel:
    return PauliceNoiseModel(gate_noise=0.012, readout_noise=READOUT_FLIP)


def aer_noise_model() -> AerNoiseModel:
    """One fixed local model used by samples and exact scores."""
    noise = AerNoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(0.003, 1), ["h", "s", "sdg", "sx", "sxdg", "t", "x", "y", "z"]
    )
    noise.add_all_qubit_quantum_error(depolarizing_error(0.025, 2), ["cx", "cz"])
    noise.add_all_qubit_readout_error(ReadoutError([[1 - READOUT_FLIP, READOUT_FLIP], [READOUT_FLIP, 1 - READOUT_FLIP]]))
    return noise


def _without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    result = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            qargs = [result.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
            result.append(instruction.operation, qargs, [])
    return result


def _operation_layers(circuit: QuantumCircuit) -> tuple[int, ...]:
    """ASAP layers for the exact instruction order used by Aer."""
    next_free = [0] * circuit.num_qubits
    layers: list[int] = []
    for instruction in circuit.data:
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        layer = max((next_free[q] for q in qubits), default=0)
        for qubit in qubits:
            next_free[qubit] = layer + 1
        layers.append(layer)
    return tuple(layers)


def _find_check_indices(circuit: QuantumCircuit) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices: list[int] = []
    for index, instruction in enumerate(circuit.data):
        qubits = {circuit.find_bit(qubit).index for qubit in instruction.qubits}
        if len(qubits) == 2 and HELPER_QUBIT in qubits and qubits & set(DATA_QUBITS):
            indices.append(index)
    if len(indices) < 2:
        raise RuntimeError("Expected Paulice to generate start and end helper interactions.")
    return tuple(indices), _operation_layers(circuit)


# q0 short/long deliberately have the same start layer and distinct endpoints.
# q1 offers two additional verified Paulice placements; its automatic candidates
# do not share a start layer, so the UI reports the actual numbers transparently.
_CHECK_SPECS: dict[int, tuple[tuple[str, int], ...]] = {
    0: (("short", 1), ("long", 3)),
    1: (("short", 1), ("long", 4)),
}


@lru_cache(maxsize=1)
def checkpoints() -> dict[int, dict[int, tuple[Checkpoint, ...]]]:
    """Build regular reference checks through the installed Paulice 0.3 API."""
    built: dict[int, dict[int, tuple[Checkpoint, ...]]] = {1: {}, 2: {}}
    for target, specs in _CHECK_SPECS.items():
        raw_options: list[tuple[str, QuantumCircuit, tuple[int, ...], tuple[int, ...], float | None]] = []
        for interval, seed in specs:
            # stabilizers='all': helper q2 alone is the syndrome (0 clear, 1 flag).
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not re-express the output.*")
                generated = add_pauli_checks(
                    _payload(), target_qubits=[target], noise_model=_paulice_search_noise(),
                    method="windowed", cost_nshots=512, seed=seed, stabilizers="all",
                )
            checked = generated[1]
            bare = _without_measurements(checked.circuit)
            if bare.num_qubits != 3 or tuple(checked.check_qubits) != (HELPER_QUBIT,):
                raise RuntimeError("This demo requires Paulice to allocate q2 as the one helper.")
            check_indices, layers = _find_check_indices(bare)
            raw_options.append((interval, bare, check_indices, layers, checked.cost))

        for level in (1, 2):
            level_options: list[Checkpoint] = []
            for interval, bare, check_indices, layers, cost in raw_options:
                circuit = _regular_level_circuit(bare, level)
                # The moved T can shift later physical layers and instruction
                # positions, so describe the circuit Aer actually receives.
                actual_indices, actual_layers = _find_check_indices(circuit)
                start, end = actual_layers[actual_indices[0]], actual_layers[actual_indices[-1]]
                level_options.append(
                    Checkpoint(
                        level=level, key=f"q{target}-{interval}", interval=interval, target=target,
                        start_layer=start, end_layer=end, separation=end - start,
                        protected_circuit=circuit, base_check_circuit=bare,
                        moved_t_indices=_moved_t_indices(circuit, level),
                        check_operation_indices=actual_indices, operation_layers=actual_layers, paulice_cost=cost,
                    )
                )
            built[level][target] = tuple(level_options)
    return built


def available_targets(level: int) -> tuple[int, ...]:
    return tuple(checkpoints()[level])


def available_checkpoints(level: int, target: int) -> tuple[Checkpoint, ...]:
    return checkpoints()[level][target]


def diagram_data(level: int, target: int, key: str) -> Checkpoint:
    return next(option for option in available_checkpoints(level, target) if option.key == key)


def _density_matrix(circuit: QuantumCircuit, noise_model: AerNoiseModel | None) -> np.ndarray:
    work = circuit.copy()
    work.save_density_matrix()
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    return np.asarray(backend.run(work).result().data(0)["density_matrix"].data, dtype=complex)


def _data_blocks_by_true_helper(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # q0/q1 are the least-significant statevector bits; q2 selects the block.
    return rho[:4, :4], rho[4:8, 4:8]


def _reported_zero_data_density(rho: np.ndarray, readout_flip: float = READOUT_FLIP) -> tuple[np.ndarray | None, float, float]:
    """Condition data on the reported helper=0 using the same readout channel as shots."""
    true_zero, true_one = _data_blocks_by_true_helper(rho)
    true_probability = float(np.real(np.trace(true_zero)))
    reported_zero = (1 - readout_flip) * true_zero + readout_flip * true_one
    probability = float(np.real(np.trace(reported_zero)))
    if probability <= 1e-12:
        return None, 0.0, true_probability
    return reported_zero / probability, probability, true_probability


def _pure_state_fidelity(rho: np.ndarray | None, ideal: np.ndarray) -> float | None:
    if rho is None:
        return None
    return float(np.clip(np.real(np.vdot(ideal, rho @ ideal)), 0.0, 1.0))


@lru_cache(maxsize=2)
def _exact_regular_values(level: int) -> tuple[float, np.ndarray]:
    """No-check values are fixed per level, independent of shots and seed."""
    ideal = _regular_ideal_state(level)
    payload = _regular_level_circuit(_payload(), level)
    return _pure_state_fidelity(_density_matrix(payload, aer_noise_model()), ideal) or 0.0, ideal


def _shot_circuit(checkpoint: Checkpoint) -> tuple[QuantumCircuit, int]:
    circuit = checkpoint.protected_circuit.copy()
    data_bits = ClassicalRegister(2, "data")
    syndrome_bit = ClassicalRegister(1, "syndrome")
    circuit.add_register(data_bits)
    circuit.add_register(syndrome_bit)
    circuit.measure(list(DATA_QUBITS), list(data_bits))
    circuit.measure([HELPER_QUBIT], list(syndrome_bit))
    return circuit, circuit.find_bit(syndrome_bit[0]).index


def _memory_bit(memory: str, clbit_index: int) -> int:
    return int(memory.replace(" ", "")[-(clbit_index + 1)])


def run_experiment(level: int, target: int, checkpoint_key: str, shots: int = 1000, seed: int = 2026) -> dict[str, Any]:
    """One independently seeded sample plus exact, seed-independent state scores."""
    option = diagram_data(level, target, checkpoint_key)
    no_check_fidelity, ideal = _exact_regular_values(level)
    rho = _density_matrix(option.protected_circuit, aer_noise_model())
    filtered_rho, predicted_acceptance, true_acceptance = _reported_zero_data_density(rho)
    filtered_fidelity = _pure_state_fidelity(filtered_rho, ideal)

    circuit, syndrome_index = _shot_circuit(option)
    result = AerSimulator(noise_model=aer_noise_model(), seed_simulator=seed).run(circuit, shots=shots, memory=True).result()
    memory = result.get_memory(circuit)
    kept = sum(_memory_bit(outcome, syndrome_index) == 0 for outcome in memory)
    return {
        "level": level, "checkpoint_key": option.key, "interval": option.interval,
        "data_qubit": f"q{option.target}", "helper": "q2", "start_layer": option.start_layer,
        "end_layer": option.end_layer, "layers": f"{option.start_layer} -> {option.end_layer}",
        "separation": option.separation, "shots": shots, "seed": seed, "kept": kept,
        "flagged": shots - kept, "sample_acceptance": kept / shots,
        "predicted_acceptance": predicted_acceptance, "true_helper_acceptance": true_acceptance,
        "no_check_fidelity": no_check_fidelity, "filtered_fidelity": filtered_fidelity,
        "fidelity_change_pp": None if filtered_fidelity is None else 100 * (filtered_fidelity - no_check_fidelity),
    }


def _insert_x(circuit: QuantumCircuit, insert_before: int, target: int) -> QuantumCircuit:
    result = QuantumCircuit(circuit.num_qubits)
    for index, instruction in enumerate(circuit.data):
        if index == insert_before:
            result.x(target)
        qargs = [result.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        result.append(instruction.operation, qargs, [])
    if insert_before == len(circuit.data):
        result.x(target)
    return result


def fault_trials() -> tuple[Checkpoint, tuple[FaultTrial, FaultTrial]]:
    """Verified caught and missed X locations on data q0, with no background noise."""
    option = available_checkpoints(1, 0)[0]
    caught = FaultTrial("Inside the check", "Immediately after the first purple helper/data interaction.", option.check_operation_indices[0] + 1, "Caught")
    missed = FaultTrial("After the check", "Immediately after the final purple helper/data interaction.", option.check_operation_indices[-1] + 1, "Missed")
    return option, (caught, missed)


def run_fault_trial(name: str) -> dict[str, Any]:
    """Execute a no-noise X-fault circuit and report its true helper result."""
    option, trials = fault_trials()
    trial = next(item for item in trials if item.name == name)
    faulty = _insert_x(option.protected_circuit, trial.insert_before, option.target)
    rho = _density_matrix(faulty, None)
    # Noise is off in this game, including measurement error.
    data_rho, helper_zero_probability, _ = _reported_zero_data_density(rho, readout_flip=0.0)
    fidelity = _pure_state_fidelity(data_rho, _regular_ideal_state(1))
    return {
        "trial": trial.name, "description": trial.description, "expected": trial.expected,
        "actual": "Missed" if helper_zero_probability > 0.5 else "Caught",
        "helper_result": "0 (clean)" if helper_zero_probability > 0.5 else "1 (flagged)",
        "helper_zero_probability": helper_zero_probability, "data_fidelity_after_fault": fidelity,
        "fault_index": trial.insert_before, "checkpoint": option,
    }


def _inside_phase_circuit(option: Checkpoint, level: int) -> QuantumCircuit:
    """Verified S/T insertion where the propagated q0 check is Z (inside its interval)."""
    if option.target != 0:
        raise ValueError("The verified inside-T pair is currently available for q0 only.")
    phase_name = "s" if level == 1 else "t"
    # Use the raw Paulice circuit so this optional matched S/T experiment remains
    # separate from the regular Level-2 moved-T payload.
    raw_indices, _ = _find_check_indices(option.base_check_circuit)
    insert_before = raw_indices[-1]
    result = QuantumCircuit(option.base_check_circuit.num_qubits)
    for index, instruction in enumerate(option.base_check_circuit.data):
        if index == insert_before:
            getattr(result, phase_name)(0)
        qargs = [result.qubits[option.base_check_circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        result.append(instruction.operation, qargs, [])
    _append_post_phase_block(result, level)
    return result


def _inside_phase_ideal(level: int) -> np.ndarray:
    """Check-free data circuit equivalent to the verified q0 inside insertion."""
    circuit = _payload()
    circuit.sx(0)
    (circuit.s if level == 1 else circuit.t)(0)
    circuit.sxdg(0)
    _append_post_phase_block(circuit, level)
    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)


def run_inside_t_comparison(interval_key: str, shots: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Matched Clifford/T runs: same q0 interval, endpoints, gate count, and noise."""
    records: list[dict[str, Any]] = []
    for level in (1, 2):
        option = diagram_data(level, 0, interval_key)
        circuit = _inside_phase_circuit(option, level)
        ideal = _inside_phase_ideal(level)
        accepted, predicted, _ = _reported_zero_data_density(_density_matrix(circuit, aer_noise_model()))
        shot_option = Checkpoint(level, option.key, option.interval, 0, option.start_layer, option.end_layer,
            option.separation, circuit, option.base_check_circuit, (), option.check_operation_indices,
            option.operation_layers, option.paulice_cost)
        shot_circuit, syndrome = _shot_circuit(shot_option)
        result = AerSimulator(noise_model=aer_noise_model(), seed_simulator=seed + level).run(shot_circuit, shots=shots, memory=True).result()
        kept = sum(_memory_bit(value, syndrome) == 0 for value in result.get_memory(shot_circuit))
        records.append({
            "level": level, "checkpoint_key": interval_key, "inside_gate": "S" if level == 1 else "T",
            "shots": shots, "seed": seed + level, "kept": kept, "sample_acceptance": kept / shots,
            "predicted_acceptance": predicted, "filtered_fidelity": _pure_state_fidelity(accepted, ideal),
        })
    return records[0], records[1]


def sampling_consistency_report(level: int = 1, target: int = 0, key: str = "q0-short", repeats: int = 24, shots: int = 1000) -> dict[str, float]:
    """Independent-seed binomial guard for the prediction/sample syndrome mapping."""
    runs = [run_experiment(level, target, key, shots, PAULICE_SEED + 100 + i) for i in range(repeats)]
    total = repeats * shots
    predicted = runs[0]["predicted_acceptance"]
    observed = sum(run["kept"] for run in runs) / total
    stderr = float(np.sqrt(predicted * (1 - predicted) / total))
    return {"prediction": predicted, "observed": observed, "pooled_shots": float(total), "binomial_stderr": stderr,
            "z_score": (observed - predicted) / stderr if stderr else 0.0}


def verification_report() -> dict[str, Any]:
    """Deterministic checks run before a demo."""
    report: dict[str, Any] = {
        "noiseless": [], "faults": [], "inside_t": [], "exact_independence": None,
        "regular_level_ideal_overlap": float(abs(np.vdot(_regular_ideal_state(1), _regular_ideal_state(2))) ** 2),
        "inside_t_ideal_overlap": float(abs(np.vdot(_inside_phase_ideal(1), _inside_phase_ideal(2))) ** 2),
    }
    for level, by_target in checkpoints().items():
        ideal = _regular_ideal_state(level)
        for options in by_target.values():
            for option in options:
                accepted, probability, _ = _reported_zero_data_density(_density_matrix(option.protected_circuit, None), 0.0)
                report["noiseless"].append({"level": level, "checkpoint": option.key, "acceptance": probability,
                                            "fidelity": _pure_state_fidelity(accepted, ideal)})
    for trial in fault_trials()[1]:
        report["faults"].append(run_fault_trial(trial.name))
    for interval in ("q0-short", "q0-long"):
        for level in (1, 2):
            option = diagram_data(level, 0, interval)
            accepted, probability, _ = _reported_zero_data_density(_density_matrix(_inside_phase_circuit(option, level), None), 0.0)
            report["inside_t"].append({"level": level, "checkpoint": interval, "acceptance": probability,
                                       "fidelity": _pure_state_fidelity(accepted, _inside_phase_ideal(level))})
    first = run_experiment(1, 0, "q0-short", 100, 7)
    second = run_experiment(1, 0, "q0-short", 2000, 999)
    report["exact_independence"] = {
        "same_prediction": abs(first["predicted_acceptance"] - second["predicted_acceptance"]) < 1e-12,
        "same_no_check_fidelity": abs(first["no_check_fidelity"] - second["no_check_fidelity"]) < 1e-12,
        "same_filtered_fidelity": abs((first["filtered_fidelity"] or 0) - (second["filtered_fidelity"] or 0)) < 1e-12,
    }
    return report
