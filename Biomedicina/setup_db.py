"""
setup_db.py
===========
Script de configuração do banco de dados SQLite para o sistema de predição de risco clínico.
Cria as tabelas necessárias e importa os dados do arquivo CSV.

Tabelas criadas:
  - usuarios  : credenciais de acesso ao sistema
  - pacientes : cadastro básico dos pacientes
  - exames    : registros históricos de exames e resultado da IA
  - tipos_exame: definição dos exames e seus limiares de risco
"""

import sqlite3
import pandas as pd
import hashlib
import os

# ──────────────────────────────────────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "clinica.db")
CSV_PATH = os.path.join(BASE_DIR, "pacientes.csv")


def hash_senha(senha: str) -> str:
    """Retorna o hash SHA-256 da senha fornecida."""
    return hashlib.sha256(senha.encode()).hexdigest()


def criar_banco():
    """Cria todas as tabelas do banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # ── Tabela de usuários ────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario  TEXT    NOT NULL UNIQUE,
            senha    TEXT    NOT NULL,
            perfil   TEXT    NOT NULL DEFAULT 'medico'
        )
    """)

    # ── Tabela de pacientes ───────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            nome      TEXT    NOT NULL,
            idade     INTEGER NOT NULL,
            criado_em TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)

    # ── Tabela de tipos de exame (configurável pelo usuário) ──────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tipos_exame (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_exame      TEXT    NOT NULL UNIQUE,
            unidade         TEXT,
            limiar_alerta   REAL,
            limiar_critico  REAL,
            descricao       TEXT
        )
    """)

    # ── Tabela de exames (resultados históricos + predição da IA) ─────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exames (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            id_paciente   INTEGER NOT NULL,
            glicose       REAL,
            pressao       REAL,
            imc           REAL,
            colesterol    REAL,
            resultado_ia  INTEGER,
            risco_label   TEXT,
            data_exame    TEXT    DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (id_paciente) REFERENCES pacientes(id)
        )
    """)

    conn.commit()
    print("[OK] Tabelas criadas com sucesso.")
    return conn


def inserir_usuarios_padrao(conn: sqlite3.Connection):
    """Insere usuários padrão caso a tabela esteja vazia."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM usuarios")
    if cur.fetchone()[0] == 0:
        usuarios = [
            ("admin",   hash_senha("admin123"),  "admin"),
            ("medico",  hash_senha("med2024"),   "medico"),
            ("enfermeiro", hash_senha("enf2024"), "medico"),
        ]
        cur.executemany(
            "INSERT INTO usuarios (usuario, senha, perfil) VALUES (?, ?, ?)",
            usuarios
        )
        conn.commit()
        print("[OK] Usuários padrão inseridos:")
        print("     admin / admin123")
        print("     medico / med2024")
        print("     enfermeiro / enf2024")
    else:
        print("[INFO] Usuários já existem — nenhuma alteração.")


def inserir_tipos_exame_padrao(conn: sqlite3.Connection):
    """Insere os tipos de exame com limiares padrão."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM tipos_exame")
    if cur.fetchone()[0] == 0:
        tipos = [
            ("Glicose",    "mg/dL", 100.0, 126.0,
             "Nível de glicose em jejum no sangue"),
            ("Pressão",    "mmHg",  130.0, 160.0,
             "Pressão arterial sistólica"),
            ("IMC",        "kg/m²",  25.0,  30.0,
             "Índice de Massa Corporal"),
            ("Colesterol", "mg/dL", 200.0, 240.0,
             "Colesterol total no sangue"),
        ]
        cur.executemany(
            """INSERT INTO tipos_exame
               (nome_exame, unidade, limiar_alerta, limiar_critico, descricao)
               VALUES (?, ?, ?, ?, ?)""",
            tipos
        )
        conn.commit()
        print("[OK] Tipos de exame padrão inseridos.")
    else:
        print("[INFO] Tipos de exame já existem — nenhuma alteração.")


def importar_csv(conn: sqlite3.Connection):
    """Importa os dados do CSV para as tabelas pacientes e exames."""
    cur = conn.cursor()

    # Verifica se já foram importados
    cur.execute("SELECT COUNT(*) FROM pacientes")
    if cur.fetchone()[0] > 0:
        print("[INFO] Dados do CSV já importados — nenhuma alteração.")
        return

    # Lê o CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    # Mapeamento de risco numérico → label
    mapa_risco = {0: "Normal", 1: "Alerta", 2: "Alto Risco"}

    total = 0
    for _, row in df.iterrows():
        # Insere paciente
        cur.execute(
            "INSERT INTO pacientes (nome, idade) VALUES (?, ?)",
            (str(row["nome"]).strip(), int(row["idade"]))
        )
        id_paciente = cur.lastrowid

        # Insere exame
        risco_num   = int(row["risco"])
        risco_label = mapa_risco.get(risco_num, "Desconhecido")
        cur.execute(
            """INSERT INTO exames
               (id_paciente, glicose, pressao, imc, colesterol,
                resultado_ia, risco_label)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                id_paciente,
                float(row["glicose"]),
                float(row["pressao"]),
                float(row["imc"]),
                float(row["colesterol"]),
                risco_num,
                risco_label,
            )
        )
        total += 1

    conn.commit()
    print(f"[OK] {total} registros importados do CSV com sucesso.")


def verificar_banco(conn: sqlite3.Connection):
    """Exibe um resumo do conteúdo do banco após a criação."""
    cur = conn.cursor()
    tabelas = ["usuarios", "pacientes", "exames", "tipos_exame"]
    print("\n── Resumo do banco de dados ──────────────────────────────")
    for tabela in tabelas:
        cur.execute(f"SELECT COUNT(*) FROM {tabela}")
        qtd = cur.fetchone()[0]
        print(f"   {tabela:<15}: {qtd:>5} registro(s)")
    print("──────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# Execução principal
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nCriando banco de dados em: {DB_PATH}\n")
    conn = criar_banco()
    inserir_usuarios_padrao(conn)
    inserir_tipos_exame_padrao(conn)
    importar_csv(conn)
    verificar_banco(conn)
    conn.close()
    print("Banco de dados configurado com sucesso!\n")
