import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analisar_fatores_transpilacao(caminho_csv="benchmark_results.csv", pasta_saida="graficos_transp"):
  
    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_csv(caminho_csv)

    colunas_analise = [
        "num_qubits", "depth_max_config", "real_depth", "num_gates",
        "num_1q_gates", "num_2q_gates", "num_cx", "num_cz", "num_swap", 
        "avg_gate_density"
    ]

    correlacoes = df[colunas_analise + ["transpile_ms"]].corr()["transpile_ms"].drop("transpile_ms")
    correlacoes_ordenadas = correlacoes.abs().sort_values(ascending=False)

    print("===== FATORES QUE MAIS IMPACTAM NA TRANSPILAÇÃO =====")
    for fator, valor in correlacoes_ordenadas.items():
        print(f"{fator:18s}: correlação = {correlacoes[fator]:+.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlacoes_ordenadas.values, y=correlacoes_ordenadas.index, palette="Blues_r")
    plt.title("Correlação com Tempo de Transpilação (ms)")
    plt.xlabel("Correlação de Pearson")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/correlacoes_transpilacao.png")
    plt.close()

analisar_fatores_transpilacao("benchmark_results.csv")
