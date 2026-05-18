"""
=============================================================
EXERCÍCIO 8: Bot Orquestrador — Roteamento de Intenções e Tool Use
=============================================================

CONCEITO CENTRAL:
  "Tool Use" (uso de ferramentas) é a capacidade de um bot decidir,
  com base na intenção detectada, qual função/API/módulo chamar.

  Isso é a base do que a OpenAI chama de "Function Calling" e
  o que a Anthropic chama de "Tool Use" nos LLMs modernos.

  Em vez de responder em texto puro, o bot age como um ORQUESTRADOR
  que delega tarefas para ferramentas especializadas.

ARQUITETURA DO ROTEADOR:
  input
    │
    ▼
  Detector de Intenção (palavras-chave / NLP)
    │
    ├── intenção: FINANCEIRO  → acionar_modulo_financeiro()
    ├── intenção: SUPORTE     → acionar_modulo_suporte()
    ├── intenção: RASTREIO    → acionar_modulo_rastreio()
    └── intenção: GERAL       → resposta_texto_padrao()

EVOLUÇÃO NATURAL:
  Este padrão é exatamente como GPT-4 / Claude funcionam com
  Function Calling / Tool Use — o LLM decide qual ferramenta
  acionar e com quais parâmetros extraídos da mensagem.
"""

import random
import time

# ------------------------------------------------------------------
# "FERRAMENTAS" — simulam chamadas a APIs reais
# ------------------------------------------------------------------

def acionar_modulo_financeiro(contexto: dict = None) -> dict:
    """
    Simula chamada à API do gateway de pagamentos.
    Em produção: chamaria Stripe, PagSeguro, Mercado Pago, etc.
    """
    time.sleep(0.5)  # simula latência de rede
    return {
        "status":           "sucesso",
        "estornos_pend":    random.randint(0, 2),
        "saldo_cashback":   round(random.uniform(0, 150), 2),
        "limite_disponivel": round(random.uniform(1000, 5000), 2),
        "fonte":            "[API] gateway-pagamentos.empresa.com",
    }

def acionar_modulo_suporte(numero_pedido: str = None) -> dict:
    """
    Simula chamada ao sistema de CRM/tickets.
    Em produção: chamaria Zendesk, Freshdesk, Salesforce, etc.
    """
    time.sleep(0.3)
    status_opcoes = ["Em rota de entrega 🚚", "Chegou ao centro de distribuição 📦", "Aguardando postagem ⏳"]
    return {
        "status":         "sucesso",
        "pedido":         numero_pedido or "#" + str(random.randint(1000, 9999)),
        "status_pedido":  random.choice(status_opcoes),
        "previsao":       "3-5 dias úteis",
        "protocolo":      f"ATD-{random.randint(100000, 999999)}",
        "fonte":          "[API] crm-interno.empresa.com",
    }

def acionar_modulo_rastreio(numero_pedido: str = None) -> dict:
    """Simula chamada à API dos Correios ou transportadora."""
    time.sleep(0.4)
    return {
        "status":         "sucesso",
        "codigo_rastreio": f"BR{random.randint(100000000, 999999999)}BR",
        "ultima_posicao": "Centro de distribuição São Paulo - SP",
        "data_evento":    "hoje às 14:32",
        "fonte":          "[API] correios.com.br/rastreio",
    }

def resposta_texto_padrao(mensagem: str) -> dict:
    """Resposta genérica quando nenhuma ferramenta é acionada."""
    return {
        "status":   "ok",
        "resposta": "Entendido. Tratando sua requisição no canal de atendimento padrão.",
        "fonte":    "[Texto] resposta-padrao",
    }


# ------------------------------------------------------------------
# ROTEADOR DE INTENÇÕES
# ------------------------------------------------------------------
REGRAS_ROTEAMENTO = [
    {
        "nome":        "FINANCEIRO",
        "keywords":    ["dinheiro", "estorno", "reembolso", "pagamento", "cashback", "pagar", "cobrado"],
        "ferramenta":  acionar_modulo_financeiro,
        "icone":       "💳",
    },
    {
        "nome":        "SUPORTE",
        "keywords":    ["problema", "quebrado", "errado", "não funciona", "defeito", "ajuda"],
        "ferramenta":  acionar_modulo_suporte,
        "icone":       "🔧",
    },
    {
        "nome":        "RASTREIO",
        "keywords":    ["rastrear", "rastreio", "onde está", "entrega", "encomenda", "chegou", "prazo"],
        "ferramenta":  acionar_modulo_rastreio,
        "icone":       "📦",
    },
]

def rotear_intencao(mensagem: str) -> tuple[str, callable]:
    """
    Detecta a intenção e retorna o nome + a função a chamar.
    Em produção: este if/else seria substituído por um LLM decidindo.
    """
    msg = mensagem.lower()
    for regra in REGRAS_ROTEAMENTO:
        if any(kw in msg for kw in regra["keywords"]):
            return regra["nome"], regra["icone"], regra["ferramenta"]
    return "GERAL", "💬", resposta_texto_padrao


def formatar_resposta_ferramenta(intencao: str, icone: str, resultado: dict) -> str:
    """Formata o resultado da ferramenta de forma amigável."""
    fonte = resultado.get("fonte", "")

    if intencao == "FINANCEIRO":
        return (
            f"{icone} Módulo Financeiro acionado!\n"
            f"   Estornos pendentes : {resultado['estornos_pend']}\n"
            f"   Saldo cashback     : R$ {resultado['saldo_cashback']:.2f}\n"
            f"   Limite disponível  : R$ {resultado['limite_disponivel']:.2f}\n"
            f"   → {fonte}"
        )
    elif intencao == "SUPORTE":
        return (
            f"{icone} Módulo Suporte acionado!\n"
            f"   Pedido    : {resultado['pedido']}\n"
            f"   Status    : {resultado['status_pedido']}\n"
            f"   Previsão  : {resultado['previsao']}\n"
            f"   Protocolo : {resultado['protocolo']}\n"
            f"   → {fonte}"
        )
    elif intencao == "RASTREIO":
        return (
            f"{icone} Módulo Rastreio acionado!\n"
            f"   Código     : {resultado['codigo_rastreio']}\n"
            f"   Localização: {resultado['ultima_posicao']}\n"
            f"   Evento     : {resultado['data_evento']}\n"
            f"   → {fonte}"
        )
    else:
        return f"{icone} {resultado['resposta']}"


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("🤖 Bot Orquestrador — Roteamento de Ferramentas")
print("   Tente falar sobre: dinheiro, estorno, rastreio,")
print("   entrega, problema, pagamento...")
print("   Digite 'sair' para encerrar")
print("=" * 55)
print("Bot: Olá! Como posso te ajudar hoje?\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Até logo! 👋")
        break

    # --- Roteamento ---
    intencao, icone, ferramenta = rotear_intencao(entrada)

    print(f"[ROTEADOR] Intenção detectada: {intencao} → chamando ferramenta...")

    # --- Execução da ferramenta ---
    resultado = ferramenta()

    # --- Formatação da resposta ---
    resposta = formatar_resposta_ferramenta(intencao, icone, resultado)
    print(f"Bot: {resposta}\n")
