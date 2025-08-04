import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analisar_csv(caminho_csv, pasta_saida="graficos"):
   
    os.makedirs(pasta_saida, exist_ok=True)

    df = pd.read_csv(caminho_csv)

    maior_transp = df["transpile_ms"].max()
    menor_transp = df["transpile_ms"].min()
    maior_exec = df["exec_ms"].max()
    menor_exec = df["exec_ms"].min()
    exec_maior = (df["exec_ms"] > df["transpile_ms"]).sum()
    transp_maior = (df["transpile_ms"] > df["exec_ms"]).sum()
    media_dif_percent = ((df["transpile_ms"] - df["exec_ms"]) / df["transpile_ms"] * 100).mean()
    

    print("===== ANÁLISE DO CSV =====")
    print(f"Maior tempo de transpilação: {maior_transp:.3f} ms")
    print(f"Menor tempo de transpilação: {menor_transp:.3f} ms")
    print(f"Maior tempo de execução    : {maior_exec:.3f} ms")
    print(f"Menor tempo de execução    : {menor_exec:.3f} ms")
    print()
    print(f"Circuitos com EXEC > TRANSPIL.: {exec_maior}")
    print(f"Circuitos com TRANSPIL. > EXEC: {transp_maior}")
    print()
    print(f"Média de diferença percentual entre execução e transpilação: {media_dif_percent:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df["transpile_ms"], bins=50, kde=True, color="skyblue")
    plt.title("Distribuição do Tempo de Transpilação")

    plt.subplot(1, 2, 2)
    sns.histplot(df["exec_ms"], bins=50, kde=True, color="salmon")
    plt.title("Distribuição do Tempo de Execução")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/histogramas_tempos.png")
    plt.close()

    colunas_corr = [
        "transpile_ms", "transpile_opt3_ms", "exec_ms", "total_ms",
        "num_qubits", "depth_max_config", "real_depth", 
        "num_gates", "num_1q_gates", "num_2q_gates", 
        "num_cx", "num_cz", "num_swap", "avg_gate_density"
    ]
    corr = df[colunas_corr].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Matriz de Correlação entre Métricas")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/correlacao_heatmap.png")
    plt.close()

    pares = [
        ("num_qubits", "transpile_ms"),
        ("num_qubits", "exec_ms"),
        ("real_depth", "exec_ms"),
        ("num_cx", "exec_ms"),
        ("avg_gate_density", "transpile_ms")
    ]

    for x, y in pares:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.3)
        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.savefig(f"{pasta_saida}/scatter_{x}_vs_{y}.png")
        plt.close()

def main():
    caminho_csv = "benchmark_results.csv"
    pasta_saida = "graficos"
    analisar_csv(caminho_csv, pasta_saida)

if __name__ == "__main__":
    main()
