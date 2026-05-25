"""
=============================================================
  TREINAMENTO E EXPORTAÇÃO DO MODELO DE RISCO CLÍNICO
=============================================================
Este script:
  1. Lê o dataset gerado anteriormente (pacientes.csv)
  2. Executa o pipeline completo de pré-processamento
  3. Treina o Random Forest (melhor modelo identificado)
  4. Gera métricas e gráficos de validação profissionais
  5. Exporta modelo + scaler + metadados em JSON
     (para uso na interface interativa)
=============================================================
"""

import json, warnings
import numpy  as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_val_score, learning_curve)
from sklearn.preprocessing    import StandardScaler, label_binarize
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix, roc_curve, auc,
                                       classification_report)

np.random.seed(42)

# ── Paleta ────────────────────────────────────
C0  = "#27ae60"   # baixo  → verde
C1  = "#e67e22"   # médio  → laranja
C2  = "#c0392b"   # alto   → vermelho
CB  = "#2c3e50"   # fundo escuro
CL  = "#ecf0f1"   # texto claro

print("=" * 62)
print("  TREINAMENTO DO MODELO — PREDIÇÃO DE RISCO CLÍNICO")
print("=" * 62)

# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
df = pd.read_csv("pacientes.csv")
FEATURES = ["idade", "glicose", "pressao", "imc", "colesterol"]
TARGET   = "risco"
ROTULOS  = {0: "Baixo", 1: "Médio", 2: "Alto"}

X = df[FEATURES].values
y = df[TARGET].values

print(f"\n  Dataset: {df.shape[0]} registros × {len(FEATURES)} features")
dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
for cls, cnt in dist.items():
    print(f"  Classe {cls} ({ROTULOS[cls]}): {cnt} ({cnt/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. SPLIT + NORMALIZAÇÃO
# ─────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ─────────────────────────────────────────────
# 3. TREINAR TODOS OS MODELOS (para comparação)
# ─────────────────────────────────────────────
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=42, n_jobs=-1),
    "KNN":                  KNeighborsClassifier(n_neighbors=7, metric="euclidean",
                                                  weights="distance"),
}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nome, mod in modelos.items():
    mod.fit(X_tr_s, y_tr)
    yp = mod.predict(X_te_s)
    cv = cross_val_score(mod, X_tr_s, y_tr, cv=kfold, scoring="accuracy")
    resultados[nome] = {
        "modelo":   mod,
        "y_pred":   yp,
        "acc":      accuracy_score(y_te, yp),
        "prec":     precision_score(y_te, yp, average="macro", zero_division=0),
        "rec":      recall_score(y_te, yp, average="macro", zero_division=0),
        "f1":       f1_score(y_te, yp, average="macro", zero_division=0),
        "cv_mean":  cv.mean(),
        "cv_std":   cv.std(),
    }
    print(f"\n  [{nome}]  Acc={resultados[nome]['acc']:.4f}  "
          f"F1={resultados[nome]['f1']:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")

# Melhor modelo = Random Forest
melhor_nome  = "Random Forest"
rf           = resultados[melhor_nome]["modelo"]
y_pred       = resultados[melhor_nome]["y_pred"]

print(f"\n  ✔ Melhor modelo: {melhor_nome}  —  Acurácia {resultados[melhor_nome]['acc']*100:.2f}%")
print("\n" + classification_report(y_te, y_pred,
      target_names=["Baixo(0)","Médio(1)","Alto(2)"], zero_division=0))

# ─────────────────────────────────────────────
# 4. LEARNING CURVE  (detecta over/underfitting)
# ─────────────────────────────────────────────
print("  Calculando learning curve...")
train_sizes, tr_scores, val_scores = learning_curve(
    rf, X_tr_s, y_tr,
    cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)
tr_mean  = tr_scores.mean(axis=1)
tr_std   = tr_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std  = val_scores.std(axis=1)

# Importância das features
feat_imp = rf.feature_importances_
feat_ord = np.argsort(feat_imp)[::-1]

# ─────────────────────────────────────────────
# 5. FIGURA MASTER — 6 painéis
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor=CB)
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def ax_style(ax, title):
    ax.set_facecolor("#34495e")
    ax.tick_params(colors=CL, labelsize=9)
    ax.xaxis.label.set_color(CL)
    ax.yaxis.label.set_color(CL)
    ax.title.set_color(CL)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a6278")

# ── Painel A: Comparação de acurácia ─────────
ax_a = fig.add_subplot(gs[0, 0])
nms  = ["Reg. Logística", "Random\nForest", "KNN"]
accs = [resultados[k]["acc"] for k in modelos]
cvs  = [resultados[k]["cv_mean"] for k in modelos]
x    = np.arange(3)
w    = 0.35
b1   = ax_a.bar(x - w/2, accs, w, color=["#3498db","#2ecc71","#e74c3c"],
                 alpha=0.9, label="Teste", zorder=3)
b2   = ax_a.bar(x + w/2, cvs,  w, color=["#3498db","#2ecc71","#e74c3c"],
                 alpha=0.45, label="CV (k=5)", zorder=3)
ax_a.set_xticks(x); ax_a.set_xticklabels(nms, fontsize=8)
ax_a.set_ylim(0.5, 1.05)
ax_a.axhline(0.5, color="#7f8c8d", lw=0.8, ls="--", alpha=0.5)
ax_a.legend(fontsize=8, facecolor="#2c3e50", labelcolor=CL)
ax_a.grid(axis="y", alpha=0.2)
for b, v in zip(b1, accs):
    ax_a.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
              ha="center", va="bottom", fontsize=8, color=CL, fontweight="bold")
ax_style(ax_a, "A — Comparação de Acurácia")

# ── Painel B: Matriz de confusão ─────────────
ax_b = fig.add_subplot(gs[0, 1])
cm   = confusion_matrix(y_te, y_pred)
im   = ax_b.imshow(cm, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
tks  = ["Baixo", "Médio", "Alto"]
ax_b.set_xticks([0,1,2]); ax_b.set_yticks([0,1,2])
ax_b.set_xticklabels(tks, color=CL, fontsize=9)
ax_b.set_yticklabels(tks, color=CL, fontsize=9)
ax_b.set_xlabel("Predito", color=CL); ax_b.set_ylabel("Real", color=CL)
thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax_b.text(j, i, str(cm[i,j]), ha="center", va="center",
                  fontsize=13, fontweight="bold",
                  color="white" if cm[i,j] > thresh else "#2c3e50")
ax_style(ax_b, "B — Matriz de Confusão (Random Forest)")

# ── Painel C: Curvas ROC ──────────────────────
ax_c = fig.add_subplot(gs[0, 2])
y_bin  = label_binarize(y_te, classes=[0,1,2])
y_prob = rf.predict_proba(X_te_s)
cores_roc = [C0, C1, C2]
nomes_cls = ["Baixo (AUC={:.3f})", "Médio (AUC={:.3f})", "Alto (AUC={:.3f})"]
for i, (cor, nm) in enumerate(zip(cores_roc, nomes_cls)):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
    ar = auc(fpr, tpr)
    ax_c.plot(fpr, tpr, color=cor, lw=2, label=nm.format(ar))
ax_c.plot([0,1],[0,1],"--", color="#7f8c8d", lw=1, alpha=0.7)
ax_c.fill_between([0,1],[0,1], alpha=0.05, color="#7f8c8d")
ax_c.set_xlabel("FPR (Falsos Positivos)"); ax_c.set_ylabel("TPR (Sensibilidade)")
ax_c.legend(fontsize=8, facecolor="#2c3e50", labelcolor=CL, loc="lower right")
ax_c.set_xlim([0,1]); ax_c.set_ylim([0,1.02])
ax_c.grid(alpha=0.15)
ax_style(ax_c, "C — Curvas ROC (One-vs-Rest)")

# ── Painel D: Learning Curve ──────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
ax_d.plot(train_sizes, tr_mean,  color="#3498db", lw=2, label="Treino")
ax_d.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.2, color="#3498db")
ax_d.plot(train_sizes, val_mean, color="#2ecc71", lw=2, label="Validação cruzada")
ax_d.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2, color="#2ecc71")
ax_d.axhline(tr_mean[-1],  color="#3498db", lw=0.8, ls=":", alpha=0.5)
ax_d.axhline(val_mean[-1], color="#2ecc71", lw=0.8, ls=":", alpha=0.5)
ax_d.set_xlabel("Amostras de treino"); ax_d.set_ylabel("Acurácia")
ax_d.set_ylim(0.5, 1.05); ax_d.grid(alpha=0.15)
ax_d.legend(fontsize=9, facecolor="#2c3e50", labelcolor=CL)
gap = tr_mean[-1] - val_mean[-1]
ax_d.annotate(f"Gap final: {gap:.4f}",
              xy=(train_sizes[-1], (tr_mean[-1]+val_mean[-1])/2),
              color="#f1c40f", fontsize=9, ha="right",
              arrowprops=dict(arrowstyle="-", color="#f1c40f", lw=0.8))
ax_style(ax_d, "D — Learning Curve (Random Forest) — diagnóstico de overfitting")

# ── Painel E: Importância das features ───────
ax_e = fig.add_subplot(gs[1, 2])
cores_feat = [C0, C1, C2, "#9b59b6", "#3498db"]
feat_nomes = [f.capitalize() for f in FEATURES]
bars = ax_e.barh([feat_nomes[i] for i in feat_ord],
                  [feat_imp[i]  for i in feat_ord],
                  color=[cores_feat[i] for i in feat_ord],
                  edgecolor="none", height=0.6)
ax_e.set_xlabel("Importância (Gini)")
for bar, val in zip(bars, [feat_imp[i] for i in feat_ord]):
    ax_e.text(val + 0.002, bar.get_y()+bar.get_height()/2,
              f"{val:.3f}", va="center", fontsize=9, color=CL, fontweight="bold")
ax_e.set_xlim(0, max(feat_imp)*1.25)
ax_e.grid(axis="x", alpha=0.15)
ax_style(ax_e, "E — Importância das Features")

# ── Painel F: Distribuição de probabilidades ──
ax_f = fig.add_subplot(gs[2, :])
probs_low  = y_prob[:, 0]
probs_mid  = y_prob[:, 1]
probs_high = y_prob[:, 2]
x_arr = np.arange(len(y_prob))
idx   = np.argsort(y_prob[:,2])     # ordena pela prob de risco alto

ax_f.bar(x_arr, y_prob[idx, 0], color=C0, label="P(Baixo)",  alpha=0.85)
ax_f.bar(x_arr, y_prob[idx, 1], color=C1, label="P(Médio)",  alpha=0.85,
         bottom=y_prob[idx, 0])
ax_f.bar(x_arr, y_prob[idx, 2], color=C2, label="P(Alto)",   alpha=0.85,
         bottom=y_prob[idx, 0] + y_prob[idx, 1])
ax_f.axhline(0.5, color="#f1c40f", lw=1, ls="--", alpha=0.6, label="Limiar 50%")
ax_f.set_xlabel("Pacientes (ordenados por P(Risco Alto))")
ax_f.set_ylabel("Probabilidade predita")
ax_f.set_xlim([0, len(y_prob)])
ax_f.set_ylim([0, 1.02])
ax_f.legend(fontsize=9, facecolor="#2c3e50", labelcolor=CL, loc="upper left", ncol=4)
ax_f.grid(axis="y", alpha=0.12)
ax_style(ax_f, "F — Distribuição de Probabilidades por Paciente (conjunto de teste)")

# Título geral
fig.text(0.5, 0.98,
         "Sistema de Predição de Risco Clínico — Análise Completa do Modelo",
         ha="center", va="top", fontsize=15, fontweight="bold", color=CL)

fig.savefig("dashboard_modelo.png", dpi=160, bbox_inches="tight", facecolor=CB)
plt.close(fig)
print("\n  ✔ dashboard_modelo.png gerado")

# ─────────────────────────────────────────────
# 6. EXPORTAR MODELO + SCALER + METADADOS
# ─────────────────────────────────────────────
# Serializa modelo e scaler com pickle (padrão da indústria)
with open("modelo_risco.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler_risco.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Metadados em JSON (consumidos pela interface HTML)
meta = {
    "model_name":   melhor_nome,
    "features":     FEATURES,
    "classes":      [0, 1, 2],
    "class_labels": ["Baixo Risco", "Risco Médio", "Risco Alto"],
    "accuracy":     round(resultados[melhor_nome]["acc"], 6),
    "f1_macro":     round(resultados[melhor_nome]["f1"],  6),
    "cv_mean":      round(resultados[melhor_nome]["cv_mean"], 6),
    "cv_std":       round(resultados[melhor_nome]["cv_std"],  6),
    "precision":    round(resultados[melhor_nome]["prec"], 6),
    "recall":       round(resultados[melhor_nome]["rec"],  6),
    "feature_importances": {
        feat: round(float(imp), 6)
        for feat, imp in zip(FEATURES, feat_imp)
    },
    "confusion_matrix": cm.tolist(),
    "scaler_mean":  [round(float(v), 4) for v in scaler.mean_],
    "scaler_std":   [round(float(v), 4) for v in scaler.scale_],
    "n_estimators": int(rf.n_estimators),
    "max_depth":    int(rf.max_depth),
    "training_samples": int(X_tr.shape[0]),
    "test_samples":     int(X_te.shape[0]),
    "auc_scores": {},
    "all_models": {
        k: {
            "accuracy":  round(v["acc"],  4),
            "f1_macro":  round(v["f1"],   4),
            "cv_mean":   round(v["cv_mean"], 4),
            "cv_std":    round(v["cv_std"],  4),
        }
        for k, v in resultados.items()
    }
}
# AUC por classe
y_bin = label_binarize(y_te, classes=[0,1,2])
for i, lbl in enumerate(["baixo","medio","alto"]):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
    meta["auc_scores"][lbl] = round(float(auc(fpr, tpr)), 6)

with open("modelo_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("  ✔ modelo_risco.pkl  exportado")
print("  ✔ scaler_risco.pkl  exportado")
print("  ✔ modelo_meta.json  exportado")
print("\n" + "=" * 62)
print(f"  MODELO PRONTO PARA PRODUÇÃO")
print(f"  Acurácia:   {meta['accuracy']*100:.2f}%")
print(f"  F1 (macro): {meta['f1_macro']:.4f}")
print(f"  AUC médio:  {np.mean(list(meta['auc_scores'].values())):.4f}")
print("=" * 62 + "\n")
