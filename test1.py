import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import zscore

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

    df['otimizacao_ganho_ms'] = df['transpile_ms'] - df['transpile_opt3_ms']
    df['otimizacao_ganho_percent'] = (df['otimizacao_ganho_ms'] / df['transpile_ms']) * 100
    print(f"\nMédia ganho otimização (ms): {df['otimizacao_ganho_ms'].mean():.3f}")
    print(f"Média ganho otimização (%): {df['otimizacao_ganho_percent'].mean():.2f}%")

    plt.figure(figsize=(8,5))
    sns.histplot(df['otimizacao_ganho_percent'], bins=50, kde=True, color='purple')
    plt.title("Distribuição do ganho percentual com otimização nível 3")
    plt.xlabel("Ganho percentual (%)")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/ganho_otimizacao_histograma.png")
    plt.close()

    df['tempo_transp_por_qubit'] = df['transpile_ms'] / df['num_qubits']
    df['tempo_exec_por_qubit'] = df['exec_ms'] / df['num_qubits']

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(df['tempo_transp_por_qubit'], bins=50, kde=True, color='skyblue')
    plt.title("Tempo de transpilação por qubit (ms)")

    plt.subplot(1,2,2)
    sns.histplot(df['tempo_exec_por_qubit'], bins=50, kde=True, color='salmon')
    plt.title("Tempo de execução por qubit (ms)")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/tempo_por_qubit.png")
    plt.close()

    df['z_transpile_ms'] = zscore(df['transpile_ms'])
    outliers = df[df['z_transpile_ms'] > 3]
    print(f"\nNúmero de outliers com z-score > 3 no tempo de transpilação: {len(outliers)}")
    if not outliers.empty:
        cols_outliers = ['transpile_ms', 'num_qubits', 'real_depth', 'avg_gate_density']
        if 'index' in df.columns:
            cols_outliers = ['index'] + cols_outliers
        print(outliers[cols_outliers])

    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='num_qubits', y='transpile_ms', alpha=0.3, label='Todos')
    sns.scatterplot(data=outliers, x='num_qubits', y='transpile_ms', color='red', label='Outliers')
    plt.title("Outliers de tempo de transpilação destacados")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/outliers_transpile.png")
    plt.close()

    df['indice_complexidade'] = df['avg_gate_density'] * df['real_depth'] * df['num_qubits']
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='indice_complexidade', y='exec_ms', alpha=0.5)
    plt.title("Tempo de execução vs Índice de complexidade")
    plt.xlabel("Índice de complexidade (avg_gate_density * real_depth * num_qubits)")
    plt.ylabel("Tempo de execução (ms)")
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/exec_vs_indice_complexidade.png")
    plt.close()

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
