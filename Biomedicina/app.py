"""
app.py
======
Sistema de Predição de Risco Clínico — Interface Streamlit
Fase 3: Integração de Banco de Dados, IA e Front-end

Funcionalidades:
  - Tela de Login com autenticação via SQLite
  - Gestão de Exames (tipos e limiares configuráveis)
  - Cadastro de Pacientes
  - Lançamento de Resultados + Predição por IA
  - Dashboard com histórico e estatísticas

Uso:
  streamlit run app.py
"""

import os
import sqlite3
import hashlib
import numpy  as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Configurações globais
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "clinica.db")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_risco.pkl")
FEATURES   = ["glicose", "pressao", "imc", "colesterol", "idade"]
CLASSES    = {0: "Normal", 1: "Alerta", 2: "Alto Risco"}

# ──────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicaIA — Predição de Risco",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS personalizado
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Cabeçalho principal */
.main-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #2980b9 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

/* Cards de risco */
.risco-normal    { background:#d5f5e3; border-left:6px solid #27ae60;
                   padding:1.2rem; border-radius:8px; }
.risco-alerta    { background:#fef9e7; border-left:6px solid #f39c12;
                   padding:1.2rem; border-radius:8px; }
.risco-alto      { background:#fadbd8; border-left:6px solid #e74c3c;
                   padding:1.2rem; border-radius:8px; }
.risco-normal h2, .risco-alerta h2, .risco-alto h2 {
    margin: 0 0 0.4rem; font-size: 1.6rem;
}
.risco-normal p, .risco-alerta p, .risco-alto p {
    margin: 0; font-size: 0.95rem;
}

/* Métricas de resumo */
.metric-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-card h3 { margin: 0; font-size: 2rem; color: #2c3e50; }
.metric-card p  { margin: 0.2rem 0 0; color: #7f8c8d; font-size: 0.9rem; }

/* Formulários */
.form-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Funções utilitárias — Banco de Dados
# ──────────────────────────────────────────────────────────────────────────────
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()


def autenticar(usuario: str, senha: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, usuario, perfil FROM usuarios WHERE usuario=? AND senha=?",
            (usuario, hash_senha(senha))
        ).fetchone()
    return dict(row) if row else None


def listar_pacientes() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT id, nome, idade, criado_em FROM pacientes ORDER BY nome",
            conn
        )


def buscar_paciente_por_id(pid: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, nome, idade FROM pacientes WHERE id=?", (pid,)
        ).fetchone()
    return dict(row) if row else None


def cadastrar_paciente(nome: str, idade: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pacientes (nome, idade) VALUES (?, ?)", (nome, idade)
        )
        conn.commit()
        return cur.lastrowid


def listar_tipos_exame() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM tipos_exame ORDER BY nome_exame", conn
        )


def salvar_tipo_exame(nome, unidade, limiar_alerta, limiar_critico, descricao):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO tipos_exame
               (nome_exame, unidade, limiar_alerta, limiar_critico, descricao)
               VALUES (?, ?, ?, ?, ?)""",
            (nome, unidade, limiar_alerta, limiar_critico, descricao)
        )
        conn.commit()


def deletar_tipo_exame(nome: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM tipos_exame WHERE nome_exame=?", (nome,))
        conn.commit()


def salvar_exame(id_paciente, glicose, pressao, imc, colesterol,
                 resultado_ia, risco_label):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO exames
               (id_paciente, glicose, pressao, imc, colesterol,
                resultado_ia, risco_label)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (id_paciente, glicose, pressao, imc, colesterol,
             resultado_ia, risco_label)
        )
        conn.commit()


def historico_exames(id_paciente: int | None = None) -> pd.DataFrame:
    with get_conn() as conn:
        if id_paciente:
            return pd.read_sql_query(
                """SELECT p.nome, e.glicose, e.pressao, e.imc, e.colesterol,
                          e.risco_label, e.data_exame
                   FROM exames e JOIN pacientes p ON p.id = e.id_paciente
                   WHERE e.id_paciente = ?
                   ORDER BY e.data_exame DESC""",
                conn, params=(id_paciente,)
            )
        return pd.read_sql_query(
            """SELECT p.nome, e.glicose, e.pressao, e.imc, e.colesterol,
                      e.risco_label, e.data_exame
               FROM exames e JOIN pacientes p ON p.id = e.id_paciente
               ORDER BY e.data_exame DESC LIMIT 200""",
            conn
        )


def estatisticas_dashboard() -> dict:
    with get_conn() as conn:
        total_pac  = conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0]
        total_exam = conn.execute("SELECT COUNT(*) FROM exames").fetchone()[0]
        dist = conn.execute(
            "SELECT risco_label, COUNT(*) as qtd FROM exames GROUP BY risco_label"
        ).fetchall()
    dist_dict = {row[0]: row[1] for row in dist}
    return {
        "total_pacientes": total_pac,
        "total_exames":    total_exam,
        "distribuicao":    dist_dict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Funções utilitárias — Modelo de IA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predizer_risco(modelo, glicose, pressao, imc, colesterol, idade):
    X = np.array([[glicose, pressao, imc, colesterol, idade]])
    pred  = modelo.predict(X)[0]
    proba = modelo.predict_proba(X)[0]
    return int(pred), proba


# ──────────────────────────────────────────────────────────────────────────────
# Componentes de UI
# ──────────────────────────────────────────────────────────────────────────────
def render_header(titulo: str, subtitulo: str = ""):
    st.markdown(f"""
    <div class="main-header">
        <h1>🏥 {titulo}</h1>
        <p>{subtitulo}</p>
    </div>
    """, unsafe_allow_html=True)


def render_card_risco(classe: int, proba: list):
    labels = {0: "Normal", 1: "Alerta", 2: "Alto Risco"}
    icones = {0: "✅", 1: "⚠️", 2: "🚨"}
    css    = {0: "risco-normal", 1: "risco-alerta", 2: "risco-alto"}
    msgs   = {
        0: "Parâmetros dentro dos limites normais. Manter acompanhamento de rotina.",
        1: "Atenção recomendada. Alguns indicadores merecem monitoramento.",
        2: "Risco elevado detectado. Avaliação médica imediata é recomendada.",
    }
    conf = proba[classe] * 100
    st.markdown(f"""
    <div class="{css[classe]}">
        <h2>{icones[classe]} Resultado: {labels[classe]}</h2>
        <p><strong>Confiança do modelo:</strong> {conf:.1f}%</p>
        <p>{msgs[classe]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Barra de probabilidades
    st.markdown("**Distribuição de probabilidades:**")
    cols = st.columns(3)
    cores = ["#27ae60", "#f39c12", "#e74c3c"]
    for i, (label, cor) in enumerate(zip(labels.values(), cores)):
        with cols[i]:
            st.metric(label=label, value=f"{proba[i]*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Páginas da aplicação
# ──────────────────────────────────────────────────────────────────────────────

# ── Tela de Login ─────────────────────────────────────────────────────────────
def pagina_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="color:#1a3c5e; font-size:2.5rem;">🏥 ClinicaIA</h1>
            <p style="color:#7f8c8d; font-size:1.1rem;">
                Sistema de Predição de Risco Clínico
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_login", clear_on_submit=False):
            st.markdown("### Acesso ao Sistema")
            usuario = st.text_input("👤 Usuário", placeholder="Digite seu usuário")
            senha   = st.text_input("🔒 Senha",   placeholder="Digite sua senha",
                                    type="password")
            submit  = st.form_submit_button("Entrar", use_container_width=True,
                                            type="primary")

        if submit:
            if not usuario or not senha:
                st.error("Por favor, preencha usuário e senha.")
            else:
                user_data = autenticar(usuario, senha)
                if user_data:
                    st.session_state["autenticado"] = True
                    st.session_state["usuario"]     = user_data["usuario"]
                    st.session_state["perfil"]      = user_data["perfil"]
                    st.session_state["pagina"]      = "dashboard"
                    st.success(f"Bem-vindo(a), {user_data['usuario']}!")
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#95a5a6; font-size:0.85rem;">
            <strong>Credenciais de demonstração:</strong><br>
            admin / admin123 &nbsp;|&nbsp; medico / med2024
        </div>
        """, unsafe_allow_html=True)


# ── Dashboard ─────────────────────────────────────────────────────────────────
def pagina_dashboard():
    render_header("Dashboard", "Visão geral do sistema")

    stats = estatisticas_dashboard()
    dist  = stats["distribuicao"]

    # Métricas principais
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Total de Pacientes", stats["total_pacientes"])
    with c2:
        st.metric("🧪 Total de Exames",    stats["total_exames"])
    with c3:
        st.metric("✅ Normal",  dist.get("Normal",    0))
    with c4:
        st.metric("⚠️ Alerta",  dist.get("Alerta",    0))

    c5, c6 = st.columns([1, 3])
    with c5:
        st.metric("🚨 Alto Risco", dist.get("Alto Risco", 0))

    st.markdown("---")

    # Últimos exames
    st.subheader("📋 Últimos Exames Registrados")
    df_hist = historico_exames()
    if not df_hist.empty:
        # Colorir coluna de risco
        def colorir_risco(val):
            cores = {"Normal": "background-color:#d5f5e3",
                     "Alerta": "background-color:#fef9e7",
                     "Alto Risco": "background-color:#fadbd8"}
            return cores.get(val, "")

        try:
            styled = df_hist.head(20).style.map(
                colorir_risco, subset=["risco_label"]
            )
        except AttributeError:
            styled = df_hist.head(20).style.applymap(
                colorir_risco, subset=["risco_label"]
            )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum exame registrado ainda.")

    # Gráficos de avaliação do modelo (se existirem)
    graficos_dir = os.path.join(BASE_DIR, "graficos")
    g1 = os.path.join(graficos_dir, "01_distribuicao_correlacao.png")
    g2 = os.path.join(graficos_dir, "02_avaliacao_modelo.png")

    if os.path.exists(g1) or os.path.exists(g2):
        st.markdown("---")
        st.subheader("📊 Avaliação do Modelo de IA")
        col_g1, col_g2 = st.columns(2)
        if os.path.exists(g1):
            with col_g1:
                st.image(g1, caption="Distribuição e Correlação", use_container_width=True)
        if os.path.exists(g2):
            with col_g2:
                st.image(g2, caption="Desempenho do Modelo", use_container_width=True)


# ── Gestão de Exames ──────────────────────────────────────────────────────────
def pagina_gestao_exames():
    render_header("Gestão de Exames", "Configure os tipos de exame e seus limiares de risco")

    df_tipos = listar_tipos_exame()

    # Tabela atual
    st.subheader("📋 Tipos de Exame Cadastrados")
    if not df_tipos.empty:
        st.dataframe(df_tipos, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum tipo de exame cadastrado.")

    st.markdown("---")

    # Formulário para adicionar/editar
    col_form, col_del = st.columns([3, 1])

    with col_form:
        st.subheader("➕ Cadastrar / Atualizar Tipo de Exame")
        with st.form("form_tipo_exame"):
            c1, c2 = st.columns(2)
            with c1:
                nome    = st.text_input("Nome do Exame *", placeholder="Ex: Glicose")
                unidade = st.text_input("Unidade",         placeholder="Ex: mg/dL")
            with c2:
                limiar_alerta  = st.number_input("Limiar de Alerta",    value=100.0, step=1.0)
                limiar_critico = st.number_input("Limiar de Alto Risco", value=126.0, step=1.0)
            descricao = st.text_area("Descrição", placeholder="Descrição do exame...")
            salvar = st.form_submit_button("💾 Salvar", type="primary",
                                           use_container_width=True)

        if salvar:
            if not nome:
                st.error("O nome do exame é obrigatório.")
            elif limiar_alerta >= limiar_critico:
                st.error("O limiar de alerta deve ser menor que o limiar crítico.")
            else:
                salvar_tipo_exame(nome, unidade, limiar_alerta, limiar_critico, descricao)
                st.success(f"Exame '{nome}' salvo com sucesso!")
                st.rerun()

    with col_del:
        st.subheader("🗑️ Remover Exame")
        if not df_tipos.empty:
            exame_del = st.selectbox("Selecione o exame",
                                     df_tipos["nome_exame"].tolist())
            if st.button("Remover", type="secondary", use_container_width=True):
                deletar_tipo_exame(exame_del)
                st.success(f"Exame '{exame_del}' removido.")
                st.rerun()

    # Tabela de referência clínica
    st.markdown("---")
    st.subheader("📚 Referência Clínica Padrão")
    referencia = pd.DataFrame({
        "Exame":          ["Glicose em Jejum", "Pressão Arterial", "IMC", "Colesterol Total"],
        "Normal":         ["< 100 mg/dL", "< 130 mmHg", "18,5 – 24,9", "< 200 mg/dL"],
        "Alerta":         ["100 – 125 mg/dL", "130 – 159 mmHg", "25 – 29,9", "200 – 239 mg/dL"],
        "Alto Risco":     ["≥ 126 mg/dL", "≥ 160 mmHg", "≥ 30", "≥ 240 mg/dL"],
        "Fonte":          ["ADA 2024", "ACC/AHA 2023", "OMS", "ACC/AHA 2023"],
    })
    st.dataframe(referencia, use_container_width=True, hide_index=True)


# ── Cadastro de Pacientes ─────────────────────────────────────────────────────
def pagina_cadastro_pacientes():
    render_header("Cadastro de Pacientes", "Registre novos pacientes no sistema")

    col_form, col_lista = st.columns([1, 2])

    with col_form:
        st.subheader("➕ Novo Paciente")
        with st.form("form_paciente", clear_on_submit=True):
            nome  = st.text_input("Nome completo *", placeholder="Ex: João da Silva")
            idade = st.number_input("Idade *", min_value=1, max_value=120, value=30, step=1)
            salvar = st.form_submit_button("💾 Cadastrar", type="primary",
                                           use_container_width=True)

        if salvar:
            if not nome.strip():
                st.error("O nome do paciente é obrigatório.")
            else:
                pid = cadastrar_paciente(nome.strip(), int(idade))
                st.success(f"Paciente **{nome}** cadastrado com ID #{pid}!")
                st.rerun()

    with col_lista:
        st.subheader("📋 Pacientes Cadastrados")
        df_pac = listar_pacientes()
        if not df_pac.empty:
            # Barra de busca
            busca = st.text_input("🔍 Buscar por nome", placeholder="Digite parte do nome...")
            if busca:
                df_pac = df_pac[df_pac["nome"].str.contains(busca, case=False, na=False)]
            st.dataframe(df_pac, use_container_width=True, hide_index=True)
            st.caption(f"Total: {len(df_pac)} paciente(s)")
        else:
            st.info("Nenhum paciente cadastrado ainda.")


# ── Lançamento de Resultados e IA ─────────────────────────────────────────────
def pagina_predicao():
    render_header("Lançamento de Exames e Predição por IA",
                  "Insira os valores dos exames e obtenha a predição de risco clínico")

    modelo = carregar_modelo()
    if modelo is None:
        st.error("⚠️ Modelo de IA não encontrado. Execute `treinar_modelo.py` primeiro.")
        return

    df_pac = listar_pacientes()
    if df_pac.empty:
        st.warning("Nenhum paciente cadastrado. Acesse 'Cadastro de Pacientes' primeiro.")
        return

    # Seleção do paciente
    st.subheader("1️⃣ Selecione o Paciente")
    opcoes = {f"{row['nome']} (ID #{row['id']}, {row['idade']} anos)": row['id']
              for _, row in df_pac.iterrows()}
    sel = st.selectbox("Paciente", list(opcoes.keys()))
    id_pac = opcoes[sel]
    pac    = buscar_paciente_por_id(id_pac)

    st.markdown("---")

    # Formulário de exames
    st.subheader("2️⃣ Informe os Valores dos Exames")

    # Buscar limiares configurados
    df_tipos = listar_tipos_exame()
    limiares = {row["nome_exame"]: row for _, row in df_tipos.iterrows()}

    with st.form("form_exame"):
        c1, c2 = st.columns(2)
        with c1:
            glicose = st.number_input(
                "🩸 Glicose (mg/dL)",
                min_value=40.0, max_value=400.0, value=100.0, step=0.1,
                help="Glicemia em jejum. Normal: < 100 mg/dL"
            )
            pressao = st.number_input(
                "💓 Pressão Arterial (mmHg)",
                min_value=60.0, max_value=250.0, value=120.0, step=0.5,
                help="Pressão sistólica. Normal: < 130 mmHg"
            )
        with c2:
            imc = st.number_input(
                "⚖️ IMC (kg/m²)",
                min_value=10.0, max_value=70.0, value=24.0, step=0.1,
                help="Índice de Massa Corporal. Normal: 18,5 – 24,9"
            )
            colesterol = st.number_input(
                "🫀 Colesterol Total (mg/dL)",
                min_value=80.0, max_value=500.0, value=180.0, step=1.0,
                help="Colesterol total. Normal: < 200 mg/dL"
            )

        st.markdown(f"**Idade do paciente:** {pac['idade']} anos *(obtida do cadastro)*")

        analisar = st.form_submit_button(
            "🤖 Analisar com Inteligência Artificial",
            type="primary", use_container_width=True
        )

    if analisar:
        st.markdown("---")
        st.subheader("3️⃣ Resultado da Predição")

        with st.spinner("Processando dados com o modelo de IA..."):
            classe, proba = predizer_risco(
                modelo, glicose, pressao, imc, colesterol, pac["idade"]
            )

        # Card de resultado
        render_card_risco(classe, proba)

        # Indicadores individuais com alertas
        st.markdown("---")
        st.subheader("📊 Análise Individual dos Indicadores")
        cols = st.columns(4)
        indicadores = [
            ("Glicose", glicose, "mg/dL",
             limiares.get("Glicose", {}).get("limiar_alerta", 100),
             limiares.get("Glicose", {}).get("limiar_critico", 126)),
            ("Pressão", pressao, "mmHg",
             limiares.get("Pressão", {}).get("limiar_alerta", 130),
             limiares.get("Pressão", {}).get("limiar_critico", 160)),
            ("IMC", imc, "kg/m²",
             limiares.get("IMC", {}).get("limiar_alerta", 25),
             limiares.get("IMC", {}).get("limiar_critico", 30)),
            ("Colesterol", colesterol, "mg/dL",
             limiares.get("Colesterol", {}).get("limiar_alerta", 200),
             limiares.get("Colesterol", {}).get("limiar_critico", 240)),
        ]
        for col, (nome_ind, valor, unid, lim_a, lim_c) in zip(cols, indicadores):
            with col:
                if valor >= lim_c:
                    st.error(f"🚨 **{nome_ind}**\n\n{valor:.1f} {unid}\n\n*Alto Risco*")
                elif valor >= lim_a:
                    st.warning(f"⚠️ **{nome_ind}**\n\n{valor:.1f} {unid}\n\n*Alerta*")
                else:
                    st.success(f"✅ **{nome_ind}**\n\n{valor:.1f} {unid}\n\n*Normal*")

        # Salvar no banco
        risco_label = CLASSES[classe]
        salvar_exame(id_pac, glicose, pressao, imc, colesterol, classe, risco_label)
        st.success(f"✅ Resultado salvo no banco de dados em {datetime.now().strftime('%d/%m/%Y %H:%M')}.")

        # Histórico do paciente
        st.markdown("---")
        st.subheader(f"📜 Histórico de Exames — {pac['nome']}")
        df_hist = historico_exames(id_pac)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("Este é o primeiro exame registrado para este paciente.")


# ── Histórico Geral ───────────────────────────────────────────────────────────
def pagina_historico():
    render_header("Histórico de Exames", "Consulte todos os exames registrados no sistema")

    df_pac = listar_pacientes()
    opcoes = {"Todos os pacientes": None}
    opcoes.update({f"{row['nome']} (ID #{row['id']})": row['id']
                   for _, row in df_pac.iterrows()})

    col_filtro, _ = st.columns([1, 2])
    with col_filtro:
        sel = st.selectbox("Filtrar por paciente", list(opcoes.keys()))
    id_filtro = opcoes[sel]

    df_hist = historico_exames(id_filtro)
    if not df_hist.empty:
        # Filtro por risco
        riscos = ["Todos"] + df_hist["risco_label"].unique().tolist()
        risco_sel = st.selectbox("Filtrar por risco", riscos)
        if risco_sel != "Todos":
            df_hist = df_hist[df_hist["risco_label"] == risco_sel]

        def colorir(val):
            cores = {"Normal": "background-color:#d5f5e3; color:#1e8449",
                     "Alerta": "background-color:#fef9e7; color:#9a7d0a",
                     "Alto Risco": "background-color:#fadbd8; color:#922b21"}
            return cores.get(val, "")

        try:
            styled = df_hist.style.map(colorir, subset=["risco_label"])
        except AttributeError:
            styled = df_hist.style.applymap(colorir, subset=["risco_label"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(f"Total de registros exibidos: {len(df_hist)}")

        # Exportar CSV
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exportar para CSV",
            data=csv,
            file_name=f"historico_exames_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Nenhum exame encontrado com os filtros selecionados.")


# ──────────────────────────────────────────────────────────────────────────────
# Barra lateral e roteamento
# ──────────────────────────────────────────────────────────────────────────────
def sidebar_menu():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <h2 style="color:#1a3c5e;">🏥 ClinicaIA</h2>
            <p style="color:#7f8c8d; font-size:0.85rem;">Sistema de Risco Clínico</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Usuário:** {st.session_state.get('usuario', '')}")
        st.markdown(f"**Perfil:** {st.session_state.get('perfil', '').capitalize()}")
        st.markdown("---")

        st.markdown("### 📌 Menu")
        paginas = {
            "dashboard":        "📊 Dashboard",
            "predicao":         "🤖 Predição por IA",
            "cadastro":         "👥 Cadastro de Pacientes",
            "gestao_exames":    "🧪 Gestão de Exames",
            "historico":        "📜 Histórico de Exames",
        }
        for key, label in paginas.items():
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.get("pagina") == key else "secondary"):
                st.session_state["pagina"] = key
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#bdc3c7; font-size:0.75rem;">
            ClinicaIA v1.0<br>
            Projeto Acadêmico — ML em Saúde<br>
            Modelo: Random Forest (93,5%)
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada principal
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Inicializar estado da sessão
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False
    if "pagina" not in st.session_state:
        st.session_state["pagina"] = "dashboard"

    # Roteamento: login ou aplicação
    if not st.session_state["autenticado"]:
        pagina_login()
        return

    # Sidebar e roteamento interno
    sidebar_menu()

    pagina = st.session_state.get("pagina", "dashboard")
    rotas  = {
        "dashboard":     pagina_dashboard,
        "predicao":      pagina_predicao,
        "cadastro":      pagina_cadastro_pacientes,
        "gestao_exames": pagina_gestao_exames,
        "historico":     pagina_historico,
    }
    rotas.get(pagina, pagina_dashboard)()


if __name__ == "__main__":
    main()
