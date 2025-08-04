import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analisar_regressao(csv_path):
    # Lê CSV
    df = pd.read_csv(csv_path)

    # Remove colunas com caminhos de arquivos
    df = df.drop(columns=["qasm_file", "img_file", "hist_file"])

    # Alvos a prever
    alvos = ["exec_ms", "transpile_ms"]

    # Features (colunas numéricas sem os alvos)
    X = df.drop(columns=alvos)

    for alvo in alvos:
        print(f"\n===== Prevendo {alvo} =====")

        y = df[alvo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Previsões
        y_pred = model.predict(X_test)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.3f} ms")
        print(f"R² : {r2:.4f}")

        # Importância das features
        importancias = pd.Series(model.feature_importances_, index=X.columns)
        importancias = importancias.sort_values(ascending=False)

        # Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(x=importancias.values, y=importancias.index, palette="viridis")
        plt.title(f"Importância das variáveis para prever {alvo}")
        plt.xlabel("Importância")
        plt.tight_layout()
        os.makedirs("graficos_modelo", exist_ok=True)
        plt.savefig(f"graficos_modelo/importancia_{alvo}.png")
        plt.close()

        print(f"Gráfico salvo em graficos_modelo/importancia_{alvo}.png")

# Exemplo de uso:
analisar_regressao("benchmark_results.csv")
