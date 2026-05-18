"""
=============================================================
EXERCÍCIO 7: RAG Primitivo — Retrieval-Augmented Generation com CSV
=============================================================

CONCEITO CENTRAL:
  RAG (Retrieval-Augmented Generation) é uma arquitetura que combina:
    1. RETRIEVAL: buscar dados relevantes de uma fonte externa
    2. AUGMENTATION: injetar esses dados no contexto
    3. GENERATION: gerar a resposta baseada nesse contexto enriquecido

  Aqui implementamos a versão mais simples de RAG:
    - Fonte de dados: arquivo CSV (em produção seria um banco vetorial)
    - Retrieval: filtro pandas por ID (em produção: busca semântica)
    - Generation: template de resposta com os dados recuperados

POR QUE RAG É IMPORTANTE?
  LLMs têm conhecimento estático (data de corte de treinamento).
  RAG conecta o modelo a dados dinâmicos e atuais em tempo real.
  É a base de todo assistente empresarial moderno.

EVOLUÇÃO DO RAG SIMPLES PARA PRODUÇÃO:
  CSV lookup → SQL query → Vector DB → Graph RAG
  (este exercício)         (avançado)   (estado da arte)
"""

import pandas as pd

# ------------------------------------------------------------------
# CLASSE: Banco de Dados de Clientes (simula o retrieval)
# ------------------------------------------------------------------
class BancoDadosClientes:
    """
    Carrega o CSV uma única vez (caching) e expõe métodos de busca.
    Em produção: seria uma conexão com PostgreSQL, MongoDB, etc.
    """

    def __init__(self, caminho_csv: str):
        self.df = pd.read_csv(caminho_csv)
        print(f"[DB] ✅ Base carregada: {len(self.df)} registros | "
              f"IDs disponíveis: {self.df['id_usuario'].min()} - {self.df['id_usuario'].max()}")

    def buscar_por_id(self, id_usuario: int) -> dict | None:
        """Retorna os dados do cliente ou None se não encontrado."""
        registro = self.df[self.df['id_usuario'] == id_usuario]
        if registro.empty:
            return None
        return registro.iloc[0].to_dict()

    def buscar_por_faixa_gasto(self, min_valor: float, max_valor: float) -> pd.DataFrame:
        """Busca clientes por faixa de valor de compras (bonus feature)."""
        return self.df[
            (self.df['historico_compras_valor'] >= min_valor) &
            (self.df['historico_compras_valor'] <= max_valor)
        ]

    def top_clientes(self, n: int = 5) -> pd.DataFrame:
        """Retorna os N clientes com maior score de satisfação."""
        return self.df.nlargest(n, 'score_satisfacao')[
            ['id_usuario', 'score_satisfacao', 'historico_compras_valor']
        ]


# ------------------------------------------------------------------
# FUNÇÃO: Geração de resposta com dados injetados (o "G" do RAG)
# ------------------------------------------------------------------
def gerar_resposta_personalizada(dados_cliente: dict) -> str:
    """
    Injeta os dados recuperados em um template de resposta.
    Isso simula o que um LLM faz quando recebe contexto via RAG.
    """
    id_usuario = dados_cliente['id_usuario']
    gasto      = dados_cliente['historico_compras_valor']
    score      = dados_cliente['score_satisfacao']
    categoria  = dados_cliente['categoria_produto']
    ultima_msg = dados_cliente['mensagem_usuario']

    # Classificação do cliente baseada no score
    if score >= 4:
        perfil = "Cliente Premium 🌟"
        oferta = "Você tem acesso ao nosso frete grátis vitalício!"
    elif score >= 3:
        perfil = "Cliente Regular 😊"
        oferta = "Que tal um cupom de 10% na sua próxima compra?"
    else:
        perfil = "Cliente em Risco ⚠️"
        oferta = "Quero resolver qualquer problema que você tenha tido."

    # Formatação monetária
    gasto_formatado = f"R$ {gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    return (
        f"📋 Dados recuperados do ID #{id_usuario}:\n"
        f"   Perfil           : {perfil}\n"
        f"   Total investido  : {gasto_formatado}\n"
        f"   Score satisfação : {score}/5 {'⭐' * score}\n"
        f"   Categoria fav.   : {categoria}\n"
        f"   Última mensagem  : '{ultima_msg[:50]}...'\n\n"
        f"💬 {oferta}"
    )


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("🗄️  Bot CRM — RAG com Consulta ao CSV")
print("   Informe um ID de usuário para recuperar seus dados")
print("   Comandos: 'top' (melhores clientes) | 'sair'")
print("=" * 55)

db = BancoDadosClientes('logs_ecommerce.csv')
print("\nBot: Por favor, informe seu ID de usuário (4 dígitos) para consulta:\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Até logo! 👋")
        break

    # --- Comando especial: top clientes ---
    if entrada.lower() == 'top':
        print("Bot: 🏆 Top 5 clientes por satisfação:")
        print(db.top_clientes(5).to_string(index=False))
        print()
        continue

    # --- Validação de input ---
    if not entrada.isdigit():
        print("Bot: Por favor, insira apenas números (ex: 1234).\n")
        continue

    # --- RETRIEVAL: busca no CSV ---
    id_usuario = int(entrada)
    dados = db.buscar_por_id(id_usuario)

    # --- GENERATION: resposta com contexto injetado ---
    if dados:
        resposta = gerar_resposta_personalizada(dados)
        print(f"Bot: {resposta}\n")
    else:
        # Sugere IDs próximos (UX melhorada)
        ids_disponiveis = db.df['id_usuario'].values
        ids_proximos = sorted(ids_disponiveis, key=lambda x: abs(x - id_usuario))[:3]
        print(f"Bot: ❌ ID #{id_usuario} não encontrado no sistema.\n"
              f"     IDs mais próximos disponíveis: {list(ids_proximos)}\n")
