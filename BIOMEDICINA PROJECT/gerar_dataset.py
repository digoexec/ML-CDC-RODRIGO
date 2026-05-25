"""
=============================================================
  GERADOR DE DATASET SINTÉTICO - PREDIÇÃO DE RISCO CLÍNICO
=============================================================
Autor: Projeto Acadêmico de Machine Learning em Saúde
Objetivo: Gerar 2000 registros biomédicos simulados com
          rótulos de risco clínico (baixo / médio / alto)
Bibliotecas: pandas, numpy
=============================================================
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. SEMENTE ALEATÓRIA (reprodutibilidade)
# ─────────────────────────────────────────────
np.random.seed(42)          # garante que os mesmos dados sejam gerados sempre
N = 2000                    # número total de registros

# ─────────────────────────────────────────────
# 2. NOMES FICTÍCIOS
# ─────────────────────────────────────────────
# Lista de nomes simples e sem sobrenomes, comuns no Brasil
nomes_masculinos = [
    "Carlos", "Lucas", "Pedro", "Marcos", "Rafael", "Bruno", "Felipe",
    "Diego", "Rodrigo", "André", "Gustavo", "Henrique", "Eduardo",
    "Leonardo", "Thiago", "Mateus", "Gabriel", "Ricardo", "Fernando",
    "Vinícius", "Igor", "Leandro", "Caio", "Fábio", "Sérgio",
    "Paulo", "João", "Jorge", "Daniel", "Alexandre"
]

nomes_femininos = [
    "Ana", "Julia", "Fernanda", "Camila", "Beatriz", "Mariana", "Leticia",
    "Patricia", "Aline", "Sabrina", "Renata", "Claudia", "Luciana",
    "Amanda", "Tatiane", "Priscila", "Vanessa", "Natalia", "Larissa",
    "Bianca", "Carla", "Sandra", "Mônica", "Simone", "Adriana",
    "Cristina", "Débora", "Elisa", "Flavia", "Giovana"
]

# Combina os dois conjuntos e sorteia aleatoriamente 2000 nomes
todos_os_nomes = nomes_masculinos + nomes_femininos
nomes = np.random.choice(todos_os_nomes, size=N, replace=True)

# ─────────────────────────────────────────────
# 3. VARIÁVEIS BIOMÉDICAS
# ─────────────────────────────────────────────

# --- IDADE ---
# Distribuição: uniforme entre 18 e 99 anos (faixa adulta completa)
idade = np.random.randint(18, 100, size=N)

# --- GLICOSE (mg/dL) ---
# Normal: 70–99 | Pré-diabético: 100–125 | Diabético: ≥ 126
# Usamos distribuição normal centrada em 105 com desvio alto para cobrir os três grupos
glicose = np.random.normal(loc=105, scale=30, size=N)
glicose = np.clip(glicose, 60, 400)   # limita a faixa fisiologicamente plausível
glicose = np.round(glicose, 1)

# --- PRESSÃO ARTERIAL SISTÓLICA (mmHg) ---
# Normal: < 120 | Elevada: 120–129 | Hipertensão estágio 1: 130–139 | Estágio 2: ≥ 140
pressao = np.random.normal(loc=128, scale=20, size=N)
pressao = np.clip(pressao, 80, 220)
pressao = np.round(pressao, 1)

# --- IMC (kg/m²) ---
# Baixo peso: < 18.5 | Normal: 18.5–24.9 | Sobrepeso: 25–29.9 | Obeso: ≥ 30
imc = np.random.normal(loc=27, scale=6, size=N)
imc = np.clip(imc, 14, 55)
imc = np.round(imc, 1)

# --- COLESTEROL TOTAL (mg/dL) ---
# Desejável: < 200 | Limítrofe: 200–239 | Alto: ≥ 240
colesterol = np.random.normal(loc=215, scale=40, size=N)
colesterol = np.clip(colesterol, 100, 400)
colesterol = np.round(colesterol, 1)

# ─────────────────────────────────────────────
# 4. REGRA PARA DEFINIÇÃO DO RISCO CLÍNICO
# ─────────────────────────────────────────────
# Cada variável contribui com pontos de risco.
# A soma determina a classe final:
#   0 → Baixo risco
#   1 → Risco médio
#   2 → Risco alto
#
# Critérios baseados em referências clínicas simplificadas:
#   Glicose ≥ 126          → +2 pontos (diabético)
#   Glicose 100–125        → +1 ponto  (pré-diabético)
#   Pressão ≥ 140          → +2 pontos (hipertensão estágio 2)
#   Pressão 130–139        → +1 ponto  (hipertensão estágio 1)
#   IMC ≥ 30               → +2 pontos (obeso)
#   IMC 25–29.9            → +1 ponto  (sobrepeso)
#   Colesterol ≥ 240       → +2 pontos (alto)
#   Colesterol 200–239     → +1 ponto  (limítrofe)
#   Idade ≥ 60             → +1 ponto  (fator etário)

def calcular_risco(glicose, pressao, imc, colesterol, idade):
    """
    Calcula o risco clínico de cada paciente com base em
    critérios biomédicos simplificados.

    Retorna:
        0 → Baixo risco  (0–2 pontos)
        1 → Risco médio  (3–4 pontos)
        2 → Risco alto   (≥ 5 pontos)
    """
    pontos = np.zeros(len(glicose), dtype=int)

    # Contribuição da glicose
    pontos += np.where(glicose >= 126, 2, np.where(glicose >= 100, 1, 0))

    # Contribuição da pressão arterial
    pontos += np.where(pressao >= 140, 2, np.where(pressao >= 130, 1, 0))

    # Contribuição do IMC
    pontos += np.where(imc >= 30, 2, np.where(imc >= 25, 1, 0))

    # Contribuição do colesterol
    pontos += np.where(colesterol >= 240, 2, np.where(colesterol >= 200, 1, 0))

    # Contribuição da idade
    pontos += np.where(idade >= 60, 1, 0)

    # Classificação final por faixas de pontuação
    risco = np.where(pontos <= 2, 0,
            np.where(pontos <= 4, 1, 2))
    return risco

# Aplica a função vetorizada em todo o dataset
risco = calcular_risco(glicose, pressao, imc, colesterol, idade)

# ─────────────────────────────────────────────
# 5. MONTAGEM DO DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "nome":     nomes,
    "idade":    idade,
    "glicose":  glicose,
    "pressao":  pressao,
    "imc":      imc,
    "colesterol": colesterol,
    "risco":    risco
})

# ─────────────────────────────────────────────
# 6. SALVAMENTO DO ARQUIVO CSV
# ─────────────────────────────────────────────
caminho_saida = "pacientes.csv"
df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
# index=False → não salva o índice numérico do DataFrame
# encoding utf-8-sig → compatível com Excel no Windows (acentuação)

# ─────────────────────────────────────────────
# 7. RELATÓRIO RÁPIDO NO TERMINAL
# ─────────────────────────────────────────────
print("=" * 55)
print("  DATASET GERADO COM SUCESSO!")
print("=" * 55)
print(f"  Arquivo salvo em: {caminho_saida}")
print(f"  Total de registros: {len(df)}")
print()
print("  Distribuição do risco clínico:")
contagem = df["risco"].value_counts().sort_index()
rotulos  = {0: "Baixo risco", 1: "Risco médio", 2: "Risco alto"}
for classe, qtd in contagem.items():
    pct = qtd / N * 100
    print(f"    {rotulos[classe]} (classe {classe}): {qtd} pacientes ({pct:.1f}%)")
print()
print("  Estatísticas das variáveis numéricas:")
print(df[["idade","glicose","pressao","imc","colesterol"]].describe().round(2))
print("=" * 55)
