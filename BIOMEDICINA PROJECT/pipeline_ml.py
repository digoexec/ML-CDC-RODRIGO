"""
=============================================================
  PIPELINE COMPLETO DE MACHINE LEARNING
  Predição de Risco Clínico com Dados Biomédicos
=============================================================
Autor: Projeto Acadêmico de ML em Saúde
Fluxo:
  1. Carregamento e exploração do dataset
  2. Pré-processamento (features, target, split, normalização)
  3. Treinamento de múltiplos modelos
  4. Avaliação completa (métricas + validação cruzada)
  5. Visualizações (comparação, matriz de confusão)
  6. Simulação de predição para novo paciente
Bibliotecas: pandas, numpy, scikit-learn, matplotlib
=============================================================
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sem janela gráfica (compatível com todos os ambientes)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")   # suprime avisos de convergência (didático)

# Pré-processamento e divisão de dados
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, label_binarize

# Modelos de classificação
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

print("\n" + "=" * 60)
print("  PIPELINE DE MACHINE LEARNING — RISCO CLÍNICO")
print("=" * 60)

# ─────────────────────────────────────────────
# ETAPA 1 — CARREGAMENTO DO DATASET
# ─────────────────────────────────────────────
print("\n[1/6] Carregando dataset...")

df = pd.read_csv("pacientes.csv")

print(f"  Registros: {df.shape[0]} linhas | {df.shape[1]} colunas")
print(f"  Colunas: {list(df.columns)}")
print("\n  Primeiras 5 linhas:")
print(df.head().to_string())
print("\n  Valores nulos por coluna:")
print(df.isnull().sum().to_string())
print("\n  Distribuição da variável alvo (risco):")
rotulos = {0: "Baixo", 1: "Médio", 2: "Alto"}
for cls, cnt in df["risco"].value_counts().sort_index().items():
    print(f"    Classe {cls} ({rotulos[cls]}): {cnt} registros ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# ETAPA 2 — SEPARAÇÃO DE FEATURES E TARGET
# ─────────────────────────────────────────────
print("\n[2/6] Separando features e target...")

# "nome" é identificador — não entra no modelo
FEATURES = ["idade", "glicose", "pressao", "imc", "colesterol"]
TARGET   = "risco"

X = df[FEATURES].values    # matriz de entrada (2000 × 5)
y = df[TARGET].values      # vetor alvo (2000,)

print(f"  X shape: {X.shape}  →  {len(FEATURES)} features")
print(f"  y shape: {y.shape}  →  classes: {np.unique(y)}")

# ─────────────────────────────────────────────
# ETAPA 3 — DIVISÃO TREINO / TESTE
# ─────────────────────────────────────────────
print("\n[3/6] Dividindo em treino (80%) e teste (20%)...")

# stratify=y → mantém a proporção de classes em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para teste = 400 registros
    random_state=42,    # reprodutibilidade
    stratify=y          # balanceia as classes
)

print(f"  Treino: {X_train.shape[0]} amostras")
print(f"  Teste:  {X_test.shape[0]} amostras")

# ─────────────────────────────────────────────
# ETAPA 4 — NORMALIZAÇÃO (StandardScaler)
# ─────────────────────────────────────────────
print("\n[4/6] Normalizando com StandardScaler...")

# StandardScaler: transforma cada feature para média=0, desvio=1
# IMPORTANTE: fit apenas no treino, transform em treino E teste
#   → evita "data leakage" (vazamento de informação do teste para o treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # aprende μ e σ no treino
X_test_scaled  = scaler.transform(X_test)         # aplica o mesmo μ e σ no teste

print("  Médias aprendidas no treino:")
for feat, mean, std in zip(FEATURES, scaler.mean_, scaler.scale_):
    print(f"    {feat:12s} → média = {mean:7.2f} | desvio = {std:6.2f}")

# ─────────────────────────────────────────────
# ETAPA 5 — DEFINIÇÃO DOS MODELOS
# ─────────────────────────────────────────────
print("\n[5/6] Definindo modelos de classificação...")

modelos = {
    # Regressão Logística: modelo linear probabilístico
    #   max_iter=1000 → mais iterações para convergência com dados normalizados
    #   C=1.0 → regularização padrão (L2)
    "Regressão Logística": LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    ),

    # Random Forest: conjunto de árvores de decisão independentes
    #   n_estimators=200 → 200 árvores (mais árvores = mais robusto, mais lento)
    #   max_depth=10 → limita profundidade para evitar overfitting
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1       # usa todos os núcleos disponíveis
    ),

    # K-Nearest Neighbors: classifica pela maioria dos k vizinhos mais próximos
    #   n_neighbors=7 → 7 vizinhos (valor ímpar evita empates em problemas binários)
    #   metric="euclidean" → distância euclidiana entre os pontos normalizados
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        metric="euclidean",
        weights="distance"   # vizinhos mais próximos têm mais peso
    ),
}

# ─────────────────────────────────────────────
# ETAPA 6 — TREINAMENTO E AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Treinando e avaliando modelos...\n")
print("-" * 60)

# Estrutura para armazenar os resultados de cada modelo
resultados = {}

# Validação cruzada estratificada (k=5)
# StratifiedKFold → mantém proporção de classes em cada fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome_modelo, modelo in modelos.items():
    print(f"\n  ▶  {nome_modelo}")
    print("  " + "─" * 45)

    # ── Treinamento ──────────────────────────
    modelo.fit(X_train_scaled, y_train)
    # O modelo aprende os padrões dos dados de treino normalizados

    # ── Predição no conjunto de teste ────────
    y_pred = modelo.predict(X_test_scaled)

    # ── Métricas de avaliação ────────────────
    # accuracy: proporção de acertos totais
    acuracia  = accuracy_score(y_test, y_pred)

    # precision (macro): média da precisão por classe (sem ponderar pelo tamanho)
    precisao  = precision_score(y_test, y_pred, average="macro", zero_division=0)

    # recall (macro): média da sensibilidade por classe
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # f1-score (macro): média harmônica de precision e recall
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # ── Validação cruzada (5-fold) ───────────
    # Avalia o modelo em 5 partições diferentes do treino → mede estabilidade
    cv_scores = cross_val_score(
        modelo, X_train_scaled, y_train,
        cv=kfold, scoring="accuracy"
    )

    # Armazena todos os resultados
    resultados[nome_modelo] = {
        "modelo":      modelo,
        "y_pred":      y_pred,
        "acuracia":    acuracia,
        "precisao":    precisao,
        "recall":      recall,
        "f1":          f1,
        "cv_media":    cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "cv_scores":   cv_scores
    }

    # Exibe métricas no terminal
    print(f"  Acurácia    no teste:   {acuracia:.4f}  ({acuracia*100:.2f}%)")
    print(f"  Precisão    (macro):    {precisao:.4f}")
    print(f"  Recall      (macro):    {recall:.4f}")
    print(f"  F1-Score    (macro):    {f1:.4f}")
    print(f"  Validação cruzada:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Scores por fold:        {np.round(cv_scores, 4)}")

    # Relatório detalhado por classe
    print("\n  Relatório por classe:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Baixo (0)", "Médio (1)", "Alto (2)"],
        zero_division=0
    ))

print("-" * 60)

# ─────────────────────────────────────────────
# COMPARAÇÃO FINAL — MELHOR MODELO
# ─────────────────────────────────────────────
print("\n  RANKING — COMPARAÇÃO POR ACURÁCIA NO TESTE:")
print("  " + "─" * 48)
print(f"  {'Modelo':<25} {'Acurácia':>10} {'F1 (macro)':>12} {'CV Média':>10}")
print("  " + "─" * 48)

ranking = sorted(resultados.items(), key=lambda x: x[1]["acuracia"], reverse=True)
for pos, (nome_m, res) in enumerate(ranking, 1):
    destaque = "  ◀ MELHOR" if pos == 1 else ""
    print(f"  {pos}. {nome_m:<22} {res['acuracia']:>10.4f} {res['f1']:>12.4f} {res['cv_media']:>10.4f}{destaque}")

melhor_nome  = ranking[0][0]
melhor_res   = ranking[0][1]
melhor_modelo = melhor_res["modelo"]

print(f"\n  ✔ Melhor modelo selecionado: {melhor_nome}")

# ─────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n  Gerando visualizações...")

# ── Paleta de cores consistente ──────────────
COR_BAIXO  = "#2ecc71"   # verde
COR_MEDIO  = "#f39c12"   # laranja
COR_ALTO   = "#e74c3c"   # vermelho
CORES_MOD  = ["#3498db", "#2ecc71", "#e74c3c"]   # azul, verde, vermelho por modelo

nomes_modelos = list(resultados.keys())
acuracias     = [resultados[n]["acuracia"]  for n in nomes_modelos]
f1s           = [resultados[n]["f1"]        for n in nomes_modelos]
cv_medias     = [resultados[n]["cv_media"]  for n in nomes_modelos]
cv_stds       = [resultados[n]["cv_std"]    for n in nomes_modelos]

# ══════════════════════════════════════════════
# FIGURA 1 — COMPARAÇÃO DE MODELOS
# ══════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("Comparação de Modelos de Classificação", fontsize=14, fontweight="bold", y=1.02)

nomes_curtos = ["Reg. Logística", "Random Forest", "KNN"]

# Gráfico 1A: Acurácia no conjunto de teste
ax = axes[0]
barras = ax.bar(nomes_curtos, acuracias, color=CORES_MOD, edgecolor="white", linewidth=1.5, width=0.5)
ax.set_title("Acurácia no Teste", fontsize=12, fontweight="bold")
ax.set_ylabel("Acurácia", fontsize=10)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Baseline (50%)")
ax.legend(fontsize=8)
# Adiciona rótulo em cima de cada barra
for barra, val in zip(barras, acuracias):
    ax.text(barra.get_x() + barra.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

# Gráfico 1B: F1-Score macro
ax = axes[1]
barras = ax.bar(nomes_curtos, f1s, color=CORES_MOD, edgecolor="white", linewidth=1.5, width=0.5)
ax.set_title("F1-Score (macro)", fontsize=12, fontweight="bold")
ax.set_ylabel("F1-Score", fontsize=10)
ax.set_ylim(0, 1.05)
for barra, val in zip(barras, f1s):
    ax.text(barra.get_x() + barra.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

# Gráfico 1C: Validação cruzada (média ± desvio)
ax = axes[2]
ax.bar(nomes_curtos, cv_medias, color=CORES_MOD, edgecolor="white", linewidth=1.5, width=0.5,
       yerr=cv_stds, capsize=6, error_kw={"linewidth": 2, "capthick": 2})
ax.set_title("Validação Cruzada (k=5)\nMédia ± Desvio Padrão", fontsize=12, fontweight="bold")
ax.set_ylabel("Acurácia Média", fontsize=10)
ax.set_ylim(0, 1.05)
for i, (val, std) in enumerate(zip(cv_medias, cv_stds)):
    ax.text(i, val + std + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
fig1.savefig("grafico_comparacao_modelos.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  ✔ Salvo: grafico_comparacao_modelos.png")

# ══════════════════════════════════════════════
# FIGURA 2 — MATRIZ DE CONFUSÃO (melhor modelo)
# ══════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(7, 6))

cm = confusion_matrix(y_test, melhor_res["y_pred"])
# A matriz de confusão mostra:
#   - Diagonal principal: predições corretas
#   - Fora da diagonal: erros (confusões entre classes)

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

classes = ["Baixo (0)", "Médio (1)", "Alto (2)"]
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Anota os valores em cada célula
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cor_texto = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=cor_texto)

ax.set_title(f"Matriz de Confusão — {melhor_nome}", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Classe Predita", fontsize=12, labelpad=10)
ax.set_ylabel("Classe Real", fontsize=12, labelpad=10)

plt.tight_layout()
fig2.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  ✔ Salvo: matriz_confusao.png")

# ══════════════════════════════════════════════
# FIGURA 3 — CURVA ROC (one-vs-rest, melhor modelo)
# ══════════════════════════════════════════════
# A curva ROC mede a capacidade discriminativa do modelo:
#   - Eixo X: taxa de falsos positivos (FPR)
#   - Eixo Y: taxa de verdadeiros positivos (TPR / recall)
#   - AUC próximo de 1.0 → excelente | 0.5 → aleatório

# Binariza o target para abordagem OvR (One-vs-Rest)
y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])     # (n_amostras, 3)

# Obtém probabilidades de cada classe (necessário para ROC)
if hasattr(melhor_modelo, "predict_proba"):
    y_prob = melhor_modelo.predict_proba(X_test_scaled)    # (n_amostras, 3)
else:
    # Para modelos sem predict_proba, usa decision_function
    y_prob = melhor_modelo.decision_function(X_test_scaled)
    # Normaliza entre 0 e 1
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

fig3, ax = plt.subplots(figsize=(7, 6))
cores_roc = [COR_BAIXO, COR_MEDIO, COR_ALTO]
nomes_classes = ["Baixo Risco", "Risco Médio", "Risco Alto"]

for i, (cor, nome_cls) in enumerate(zip(cores_roc, nomes_classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    area = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=cor, linewidth=2.5,
            label=f"{nome_cls} (AUC = {area:.3f})")

# Linha de referência (modelo aleatório)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Aleatório (AUC = 0.500)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

ax.set_title(f"Curva ROC (One-vs-Rest) — {melhor_nome}", fontsize=13, fontweight="bold")
ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(alpha=0.3)

plt.tight_layout()
fig3.savefig("curva_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  ✔ Salvo: curva_roc.png")

# ─────────────────────────────────────────────
# ETAPA FINAL — PREDIÇÃO PARA NOVO PACIENTE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SIMULAÇÃO — PREDIÇÃO PARA NOVO PACIENTE")
print("=" * 60)

# Dados do paciente simulado (valores preenchidos manualmente)
novo_paciente = {
    "nome":       "Marina",
    "idade":       62,
    "glicose":    138.5,    # glicose elevada (diabético)
    "pressao":    145.0,    # hipertensão estágio 2
    "imc":         31.2,    # obesidade grau I
    "colesterol": 248.0,    # colesterol alto
}

print(f"\n  Paciente: {novo_paciente['nome']}")
print(f"  {'Variável':<15} {'Valor':>8}  {'Referência'}")
print("  " + "─" * 52)
referencias = {
    "idade":      "18–99 anos",
    "glicose":    "< 100 mg/dL (normal)",
    "pressao":    "< 120 mmHg (normal)",
    "imc":        "18.5–24.9 (normal)",
    "colesterol": "< 200 mg/dL (desejável)",
}
for feat in FEATURES:
    print(f"  {feat:<15} {novo_paciente[feat]:>8}  {referencias[feat]}")

# Monta o array de entrada para o modelo
X_novo = np.array([[
    novo_paciente["idade"],
    novo_paciente["glicose"],
    novo_paciente["pressao"],
    novo_paciente["imc"],
    novo_paciente["colesterol"]
]])

# Normaliza com o mesmo scaler ajustado no treino
X_novo_scaled = scaler.transform(X_novo)

# Predição de classe e probabilidades
classe_pred  = melhor_modelo.predict(X_novo_scaled)[0]
probs        = melhor_modelo.predict_proba(X_novo_scaled)[0]

# ── Resultado formatado ──────────────────────
rotulos_risco = {
    0: ("BAIXO RISCO",  "✅", "Manter hábitos saudáveis e monitoramento anual"),
    1: ("RISCO MÉDIO",  "⚠️ ", "Consulta médica semestral e ajuste do estilo de vida"),
    2: ("RISCO ALTO",   "🚨", "Consulta imediata e investigação clínica urgente"),
}

nome_risco, emoji, recomendacao = rotulos_risco[classe_pred]

print(f"\n  {'─'*52}")
print(f"  RESULTADO DA PREDIÇÃO ({melhor_nome})")
print(f"  {'─'*52}")
print(f"  Classificação:   {emoji}  {nome_risco}  (classe {classe_pred})")
print(f"\n  Probabilidades por classe:")
print(f"    Baixo Risco  (0):  {probs[0]*100:6.2f}%  {'█' * int(probs[0]*30)}")
print(f"    Risco Médio  (1):  {probs[1]*100:6.2f}%  {'█' * int(probs[1]*30)}")
print(f"    Risco Alto   (2):  {probs[2]*100:6.2f}%  {'█' * int(probs[2]*30)}")
print(f"\n  Recomendação:  {recomendacao}")
print(f"  {'─'*52}")

# ─────────────────────────────────────────────
# RESUMO FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE CONCLUÍDO — ARQUIVOS GERADOS")
print("=" * 60)
print("  • grafico_comparacao_modelos.png")
print("  • matriz_confusao.png")
print("  • curva_roc.png")
print(f"\n  Melhor modelo: {melhor_nome}")
print(f"  Acurácia:      {melhor_res['acuracia']*100:.2f}%")
print(f"  F1-Score:      {melhor_res['f1']:.4f}")
print("=" * 60 + "\n")
