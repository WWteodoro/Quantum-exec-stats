#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import csv
import time
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import (
    HGate, XGate, YGate, ZGate,
    SGate, TGate, CXGate, CZGate, SwapGate
)
from qiskit.visualization import plot_histogram


N = 50000     
OUTPUT_DIR = "resultados"


os.makedirs(f"{OUTPUT_DIR}/imgs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/hist", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/qasm", exist_ok=True)

S_g = [HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CZGate, SwapGate]

def aquecer_sistema(sim):
    print("Aquecendo sistema...")
    dummy = QuantumCircuit(3)
    dummy.h(0)
    dummy.cx(0, 1)
    dummy.measure_all()
    transpile(dummy, sim, optimization_level=3)
    sim.run(transpile(dummy, sim)).result()
    print("Sistema aquecido.")

def generate_random_circuit(W, D, S_g, max_ops=2000, min_ops=10):
    qc = QuantumCircuit(W)
    depths = [0] * W
    ops = 0

    while ops < max_ops:
        g_class = random.choice(S_g)
        gate = g_class()

        if gate.num_qubits == 1:
            q = random.randint(0, W - 1)
            qc.append(gate, [q])
            depths[q] += 1
        elif gate.num_qubits == 2 and W >= 2:
            q0, q1 = random.sample(range(W), 2)
            qc.append(gate, [q0, q1])
            d = max(depths[q0], depths[q1])
            depths[q0] = depths[q1] = d + 1
        else:
            continue  

        ops += 1

        if max(depths) >= D and ops >= min_ops:
            break

    qc.measure_all()
    return qc

def save_qasm_compat(qc, filepath):
    try:
        qasm_str = qc.qasm()
    except AttributeError:
        qasm_str = "// QASM não suportado nesta versão do Qiskit\n"
        qasm_str += "// Atualize sua biblioteca para suportar a exportação de QASM.\n"
    with open(filepath, "w") as f:
        f.write(qasm_str)


def main():
    sim = AerSimulator()
    aquecer_sistema(sim)

    best_transp = (None, float('inf'))
    worst_transp = (None, 0.0)
    best_exec = (None, float('inf'))
    worst_exec = (None, 0.0)

    csv_file = open(f"{OUTPUT_DIR}/benchmark_results.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "index", "transpile_ms", "transpile_opt3_ms", "exec_ms", "total_ms",
        "num_qubits", "depth_max_config", "real_depth", "num_gates",
        "num_1q_gates", "num_2q_gates", "num_cx", "num_cz", "num_swap",
        "avg_gate_density", "qasm_file", "img_file", "hist_file"
    ])

    random.seed(time.time_ns())
    for i in range(N):
      
        num_qubits = random.randint(8, 20)
        depth_max = random.randint(8, 20)
        qc = generate_random_circuit(num_qubits, depth_max, S_g)

     
        t0 = time.perf_counter()
        tc = transpile(qc, backend=sim, optimization_level=0)
        t1 = time.perf_counter()
        transp_ms = (t1 - t0) * 1000

        t2 = time.perf_counter()
        tc_opt = transpile(qc, backend=sim, optimization_level=3)
        t3 = time.perf_counter()
        transp_opt_ms = (t3 - t2) * 1000

        
        t4 = time.perf_counter()
        job = sim.run(tc_opt, shots=1024)
        result = job.result()
        t5 = time.perf_counter()
        exec_ms = (t5 - t4) * 1000

        total_ms = transp_ms + exec_ms

        
        depth_real = qc.depth()
        num_gates = len(qc.data)
        counts = qc.count_ops()
        num_cx = counts.get("cx", 0)
        num_cz = counts.get("cz", 0)
        num_swap = counts.get("swap", 0)
        num_1q = sum(counts.get(g, 0) for g in ["h", "x", "y", "z", "s", "t"])
        num_2q = num_cx + num_cz + num_swap
        avg_gate_density = num_gates / depth_real if depth_real > 0 else 0

    
        img_path = f"{OUTPUT_DIR}/imgs/circuit_{i}.png"
        hist_path = f"{OUTPUT_DIR}/hist/hist_{i}.png"
        qasm_path = f"{OUTPUT_DIR}/qasm/circuit_{i}.qasm"

        
        qc.draw('mpl')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        plot_histogram(result.get_counts())
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        save_qasm_compat(qc, qasm_path)

        writer.writerow([
            i,
            f"{transp_ms:.3f}",
            f"{transp_opt_ms:.3f}",
            f"{exec_ms:.3f}",
            f"{total_ms:.3f}",
            num_qubits,
            depth_max,
            depth_real,
            num_gates,
            num_1q,
            num_2q,
            num_cx,
            num_cz,
            num_swap,
            f"{avg_gate_density:.4f}",
            qasm_path,
            img_path,
            hist_path
        ])

        print(f"\n[Circuito {i}]")
        print(qc.draw(output="text"))

    csv_file.close()

    print("\n==== RESULTADOS FINAIS ====")
    print(f"Transp. mais rápida: Circuito {best_transp[0]} - {best_transp[1]:.3f} ms")
    print(f"Transp. mais lenta : Circuito {worst_transp[0]} - {worst_transp[1]:.3f} ms")
    print(f"Exec. mais rápida  : Circuito {best_exec[0]} - {best_exec[1]:.3f} ms")
    print(f"Exec. mais lenta   : Circuito {worst_exec[0]} - {worst_exec[1]:.3f} ms")

if __name__ == "__main__":
    main()

