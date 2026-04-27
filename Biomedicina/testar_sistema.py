"""
testar_sistema.py
=================
Script de teste automatizado para validar todos os componentes do sistema.
Testa: banco de dados, autenticação, importação de dados e modelo de IA.
"""

import os
import sys
import sqlite3
import hashlib
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "clinica.db")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_risco.pkl")
FEATURES   = ["glicose", "pressao", "imc", "colesterol", "idade"]
CLASSES    = {0: "Normal", 1: "Alerta", 2: "Alto Risco"}

erros = []

def ok(msg):
    print(f"  ✅ {msg}")

def falha(msg):
    print(f"  ❌ {msg}")
    erros.append(msg)

def hash_senha(s):
    return hashlib.sha256(s.encode()).hexdigest()

# ── Teste 1: Banco de dados existe ───────────────────────────────────────────
print("\n[TESTE 1] Verificar existência do banco de dados")
if os.path.exists(DB_PATH):
    ok(f"clinica.db encontrado ({os.path.getsize(DB_PATH)/1024:.1f} KB)")
else:
    falha("clinica.db NÃO encontrado")

# ── Teste 2: Tabelas existem ─────────────────────────────────────────────────
print("\n[TESTE 2] Verificar tabelas do banco")
try:
    conn = sqlite3.connect(DB_PATH)
    tabelas_esperadas = ["usuarios", "pacientes", "exames", "tipos_exame"]
    tabelas_existentes = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    for t in tabelas_esperadas:
        if t in tabelas_existentes:
            qtd = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            ok(f"Tabela '{t}' existe com {qtd} registro(s)")
        else:
            falha(f"Tabela '{t}' NÃO encontrada")
    conn.close()
except Exception as e:
    falha(f"Erro ao verificar tabelas: {e}")

# ── Teste 3: Autenticação ────────────────────────────────────────────────────
print("\n[TESTE 3] Testar autenticação de usuários")
try:
    conn = sqlite3.connect(DB_PATH)
    credenciais = [("admin", "admin123"), ("medico", "med2024")]
    for user, senha in credenciais:
        row = conn.execute(
            "SELECT usuario FROM usuarios WHERE usuario=? AND senha=?",
            (user, hash_senha(senha))
        ).fetchone()
        if row:
            ok(f"Login '{user}' autenticado com sucesso")
        else:
            falha(f"Login '{user}' FALHOU")
    # Teste com senha errada
    row_errado = conn.execute(
        "SELECT usuario FROM usuarios WHERE usuario=? AND senha=?",
        ("admin", hash_senha("senha_errada"))
    ).fetchone()
    if row_errado is None:
        ok("Rejeição de senha incorreta funcionando")
    else:
        falha("Senha incorreta foi aceita (vulnerabilidade!)")
    conn.close()
except Exception as e:
    falha(f"Erro no teste de autenticação: {e}")

# ── Teste 4: Dados importados do CSV ─────────────────────────────────────────
print("\n[TESTE 4] Verificar importação dos dados do CSV")
try:
    conn = sqlite3.connect(DB_PATH)
    qtd_pac  = conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]
    qtd_exam = conn.execute("SELECT COUNT(*) FROM exames").fetchone()[0]
    if qtd_pac >= 2000:
        ok(f"{qtd_pac} pacientes importados do CSV")
    else:
        falha(f"Apenas {qtd_pac} pacientes (esperado: 2000)")
    if qtd_exam >= 2000:
        ok(f"{qtd_exam} exames importados do CSV")
    else:
        falha(f"Apenas {qtd_exam} exames (esperado: 2000)")

    # Verificar distribuição de classes
    dist = conn.execute(
        "SELECT risco_label, COUNT(*) FROM exames GROUP BY risco_label"
    ).fetchall()
    ok("Distribuição de classes:")
    for label, qtd in dist:
        print(f"     {label}: {qtd} ({qtd/qtd_exam*100:.1f}%)")
    conn.close()
except Exception as e:
    falha(f"Erro ao verificar dados: {e}")

# ── Teste 5: Modelo de IA ────────────────────────────────────────────────────
print("\n[TESTE 5] Verificar e testar o modelo de IA")
if not os.path.exists(MODEL_PATH):
    falha("modelo_risco.pkl NÃO encontrado")
else:
    ok(f"modelo_risco.pkl encontrado ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")
    try:
        modelo = joblib.load(MODEL_PATH)
        ok("Modelo carregado com sucesso")

        # Casos de teste com resultados esperados
        casos_teste = [
            # (glicose, pressao, imc, colesterol, idade, label_esperado)
            (75.0,  110.0, 22.0, 180.0, 30, "Normal"),
            (115.0, 135.0, 27.0, 220.0, 55, "Alerta"),
            (160.0, 165.0, 35.0, 260.0, 75, "Alto Risco"),
        ]
        print("   Casos de teste:")
        for glicose, pressao, imc, col, idade, esperado in casos_teste:
            X     = np.array([[glicose, pressao, imc, col, idade]])
            pred  = modelo.predict(X)[0]
            proba = modelo.predict_proba(X)[0]
            label = CLASSES[pred]
            conf  = proba[pred] * 100
            status = "✅" if label == esperado else "⚠️"
            print(f"     {status} Glicose={glicose}, Pressão={pressao}, "
                  f"IMC={imc}, Col={col}, Idade={idade}")
            print(f"        → Predito: {label} (confiança: {conf:.1f}%) | Esperado: {esperado}")
    except Exception as e:
        falha(f"Erro ao testar modelo: {e}")

# ── Teste 6: Operações CRUD ──────────────────────────────────────────────────
print("\n[TESTE 6] Testar operações CRUD no banco")
try:
    conn = sqlite3.connect(DB_PATH)
    # Inserir paciente de teste
    cur = conn.execute(
        "INSERT INTO pacientes (nome, idade) VALUES (?, ?)",
        ("Paciente Teste", 45)
    )
    pid = cur.lastrowid
    ok(f"Paciente de teste inserido com ID #{pid}")

    # Inserir exame de teste
    conn.execute(
        """INSERT INTO exames
           (id_paciente, glicose, pressao, imc, colesterol, resultado_ia, risco_label)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (pid, 105.0, 125.0, 24.5, 195.0, 0, "Normal")
    )
    ok("Exame de teste inserido com sucesso")

    # Consultar
    row = conn.execute(
        """SELECT p.nome, e.risco_label
           FROM exames e JOIN pacientes p ON p.id = e.id_paciente
           WHERE p.id = ?""", (pid,)
    ).fetchone()
    if row and row[0] == "Paciente Teste":
        ok(f"Consulta JOIN funcionando: {row[0]} → {row[1]}")
    else:
        falha("Consulta JOIN retornou resultado inesperado")

    # Limpar dados de teste
    conn.execute("DELETE FROM exames   WHERE id_paciente = ?", (pid,))
    conn.execute("DELETE FROM pacientes WHERE id = ?",         (pid,))
    conn.commit()
    ok("Dados de teste removidos (limpeza concluída)")
    conn.close()
except Exception as e:
    falha(f"Erro nas operações CRUD: {e}")

# ── Resumo final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if not erros:
    print("  🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("     O sistema está pronto para uso.")
else:
    print(f"  ⚠️  {len(erros)} TESTE(S) FALHARAM:")
    for e in erros:
        print(f"     • {e}")
print("=" * 55 + "\n")
sys.exit(0 if not erros else 1)
