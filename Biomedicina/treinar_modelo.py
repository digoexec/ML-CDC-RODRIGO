"""
treinar_modelo.py
=================
Script de treinamento do modelo de Machine Learning para predição de risco clínico.

Pipeline:
  1. Carrega os dados do banco SQLite (tabela exames + pacientes)
  2. Realiza análise exploratória e pré-processamento
  3. Treina um RandomForestClassifier com validação cruzada
  4. Avalia o modelo com métricas completas
  5. Salva o modelo treinado em modelo_risco.pkl
  6. Gera gráficos de avaliação

Classes de saída:
  0 → Normal
  1 → Alerta
  2 → Alto Risco
"""

import os
import sqlite3
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configurações de caminhos
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "clinica.db")
MODEL_PATH  = os.path.join(BASE_DIR, "modelo_risco.pkl")
GRAFICOS_DIR = os.path.join(BASE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

FEATURES = ["glicose", "pressao", "imc", "colesterol", "idade"]
TARGET   = "resultado_ia"
CLASSES  = ["Normal", "Alerta", "Alto Risco"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Carregamento dos dados
# ──────────────────────────────────────────────────────────────────────────────
def carregar_dados() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT p.idade,
               e.glicose, e.pressao, e.imc, e.colesterol,
               e.resultado_ia
        FROM   exames    e
        JOIN   pacientes p ON p.id = e.id_paciente
        WHERE  e.resultado_ia IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"[OK] Dados carregados: {len(df)} registros, {df.shape[1]} colunas")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Análise exploratória
# ──────────────────────────────────────────────────────────────────────────────
def analise_exploratoria(df: pd.DataFrame):
    print("\n── Distribuição das classes ──────────────────────────────")
    dist = df[TARGET].value_counts().sort_index()
    for idx, cnt in dist.items():
        print(f"   Classe {idx} ({CLASSES[idx]}): {cnt:>5} ({cnt/len(df)*100:.1f}%)")

    print("\n── Estatísticas descritivas ──────────────────────────────")
    print(df[FEATURES].describe().round(2).to_string())

    # Gráfico de distribuição das classes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cores = ["#2ecc71", "#f39c12", "#e74c3c"]

    dist.plot(kind="bar", ax=axes[0], color=cores, edgecolor="black")
    axes[0].set_title("Distribuição das Classes de Risco", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Quantidade")
    axes[0].set_xticklabels(CLASSES, rotation=0)

    # Correlação
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("Mapa de Correlação entre Variáveis", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, "01_distribuicao_correlacao.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Gráfico salvo: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Pré-processamento
# ──────────────────────────────────────────────────────────────────────────────
def preprocessar(df: pd.DataFrame):
    df = df.dropna()
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 4. Treinamento e avaliação
# ──────────────────────────────────────────────────────────────────────────────
def treinar_avaliar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: escalonamento + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    print(f"\n── Validação Cruzada (5-fold) ────────────────────────────")
    print(f"   Acurácia média : {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"   Scores por fold: {np.round(scores, 4)}")

    # Treinamento final
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Avaliação no Conjunto de Teste ────────────────────────")
    print(f"   Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    print("\n── Relatório de Classificação ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    return pipeline, X_test, y_test, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# 5. Gráficos de avaliação
# ──────────────────────────────────────────────────────────────────────────────
def gerar_graficos_avaliacao(pipeline, X_test, y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cores = ["#2ecc71", "#f39c12", "#e74c3c"]

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Matriz de Confusão", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")

    # Importância das features
    rf_model  = pipeline.named_steps["clf"]
    importancias = rf_model.feature_importances_
    idx_sorted   = np.argsort(importancias)[::-1]
    features_sorted = [FEATURES[i] for i in idx_sorted]
    imp_sorted      = importancias[idx_sorted]

    bars = axes[1].barh(features_sorted[::-1], imp_sorted[::-1],
                        color=cores[2], edgecolor="black")
    axes[1].set_title("Importância das Variáveis (Feature Importance)",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Importância Relativa")
    for bar, val in zip(bars, imp_sorted[::-1]):
        axes[1].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, "02_avaliacao_modelo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico salvo: {path}")

    # Gráfico de distribuição das predições vs real
    fig2, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASSES))
    width = 0.35
    real_counts = [np.sum(y_test == i) for i in range(3)]
    pred_counts = [np.sum(y_pred == i) for i in range(3)]
    ax.bar(x - width/2, real_counts, width, label="Real",    color="#3498db", edgecolor="black")
    ax.bar(x + width/2, pred_counts, width, label="Predito", color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_title("Distribuição Real vs. Predita (Conjunto de Teste)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Quantidade")
    ax.legend()
    plt.tight_layout()
    path2 = os.path.join(GRAFICOS_DIR, "03_real_vs_predito.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico salvo: {path2}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Salvar modelo
# ──────────────────────────────────────────────────────────────────────────────
def salvar_modelo(pipeline):
    joblib.dump(pipeline, MODEL_PATH)
    tamanho = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n[OK] Modelo salvo em: {MODEL_PATH}")
    print(f"     Tamanho: {tamanho:.1f} KB")


# ──────────────────────────────────────────────────────────────────────────────
# Execução principal
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TREINAMENTO DO MODELO DE PREDIÇÃO DE RISCO CLÍNICO")
    print("=" * 60)

    df                          = carregar_dados()
    analise_exploratoria(df)
    X, y                        = preprocessar(df)
    pipeline, X_test, y_test, y_pred = treinar_avaliar(X, y)
    gerar_graficos_avaliacao(pipeline, X_test, y_test, y_pred)
    salvar_modelo(pipeline)

    print("\n" + "=" * 60)
    print("  TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("  Arquivo gerado: modelo_risco.pkl")
    print("  Features esperadas pelo modelo:", FEATURES)
    print("=" * 60 + "\n")
