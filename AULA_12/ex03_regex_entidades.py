"""
=============================================================
EXERCÍCIO 3: Extração de Entidades via Regex no Fluxo de Atendimento
=============================================================

CONCEITO CENTRAL:
  Expressões Regulares (Regex) são padrões que descrevem formatos
  de texto. Usamos o módulo `re` do Python para encontrar dados
  estruturados dentro de texto livre (não estruturado).

  Padrão usado: r'#\d{4}'
    #     → o símbolo cerquilha/hashtag literalmente
    \d    → qualquer dígito (0-9)
    {4}   → exatamente 4 repetições
    
  Resultado: encontra #1234, #0001, #9999, etc.

  Isso se chama "Extração de Entidades Nomeadas" (NER simplificado).
  Em sistemas avançados, modelos como spaCy ou BERT fazem isso
  automaticamente para qualquer tipo de entidade.

OUTROS PADRÕES ÚTEIS PARA CHATBOTS:
  CPF:    r'\d{3}\.\d{3}\.\d{3}-\d{2}'
  Email:  r'[\w\.-]+@[\w\.-]+\.\w{2,}'
  CEP:    r'\d{5}-\d{3}'
  Fone:   r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}'
"""

import re

# ------------------------------------------------------------------
# FUNÇÕES DE EXTRAÇÃO DE ENTIDADES
# ------------------------------------------------------------------
def extrair_numero_pedido(texto: str) -> list[str]:
    """Retorna todos os códigos de pedido (#XXXX) encontrados no texto."""
    return re.findall(r'#\d{4}', texto)

def extrair_cpf(texto: str) -> list[str]:
    """Extrai CPFs no formato 000.000.000-00."""
    return re.findall(r'\d{3}\.\d{3}\.\d{3}-\d{2}', texto)

def extrair_email(texto: str) -> list[str]:
    """Extrai endereços de email."""
    return re.findall(r'[\w\.-]+@[\w\.-]+\.\w{2,}', texto)

def analisar_mensagem(texto: str) -> dict:
    """
    Roda todos os extratores e retorna um relatório de entidades.
    Esse padrão se chama 'pipeline de NLP'.
    """
    return {
        "pedidos":  extrair_numero_pedido(texto),
        "cpfs":     extrair_cpf(texto),
        "emails":   extrair_email(texto),
        "original": texto,
    }

def gerar_resposta(entidades: dict) -> str:
    """Gera a resposta do bot baseada nas entidades extraídas."""
    pedidos = entidades["pedidos"]
    emails  = entidades["emails"]
    cpfs    = entidades["cpfs"]

    partes = []

    if pedidos:
        codigos = ", ".join(pedidos)
        partes.append(
            f"✅ Pedido(s) encontrado(s): {codigos}. "
            f"Iniciando consulta no sistema..."
        )
        # Simula status diferentes por pedido
        for i, pedido in enumerate(pedidos):
            status = ["Em rota de entrega 🚚", "Processando pagamento 💳", "Aguardando postagem 📦"][i % 3]
            partes.append(f"   → {pedido}: {status}")

    if emails:
        partes.append(f"📧 Email(s) capturado(s): {', '.join(emails)}. Enviaremos confirmação em breve.")

    if cpfs:
        # Mascaramos por segurança (boa prática!)
        cpfs_mascarados = [f"***{cpf[-6:]}" for cpf in cpfs]
        partes.append(f"🔒 CPF(s) detectado(s) e mascarado(s): {', '.join(cpfs_mascarados)}.")

    if not partes:
        return (
            "⚠️  Não consegui identificar um número de pedido válido.\n"
            "   Use o formato #1234 (cerquilha + 4 dígitos).\n"
            "   Exemplo: 'Meu pedido #4521 não chegou'"
        )

    return "\n".join(partes)


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("🔍 Bot de Triagem com Extração de Entidades (Regex)")
print("   Formatos aceitos:")
print("   • Pedido : #1234")
print("   • CPF    : 000.000.000-00")
print("   • Email  : usuario@email.com")
print("   Digite 'sair' para encerrar")
print("=" * 55)
print("Bot: Olá! Descreva seu problema e informe o número do pedido (Ex: #1234).\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Atendimento encerrado. Até logo! 👋")
        break

    # Pipeline de extração
    entidades = analisar_mensagem(entrada)
    resposta  = gerar_resposta(entidades)

    print(f"Bot: {resposta}\n")
