"""
=============================================================
EXERCÍCIO 4: Análise de Sentimento com Gatilho para Transbordo Humano
=============================================================

CONCEITO CENTRAL:
  Análise de Sentimento avalia a carga emocional de um texto.
  Aqui usamos abordagem léxica (dicionário de palavras) — a versão
  mais simples. Versões avançadas usam modelos como VADER, TextBlob
  ou transformers (BERT) para capturar nuance e contexto.

PONTUAÇÃO LÉXICA:
  Criamos um dicionário com pesos por palavra:
    "péssimo"   = -2 (fortemente negativo)
    "atrasou"   = -1 (levemente negativo)
    "obrigado"  = +2 (positivo)

  Score total = soma dos pesos → define a ação do bot.

TRANSBORDO (Escalation / Handoff):
  Quando o cliente está muito irritado, o bot automaticamente
  transfere para um humano. Isso é chamado de "Human-in-the-Loop".
  É uma salvaguarda crítica em sistemas de produção.

NÍVEIS DE IRRITAÇÃO:
  0     → Normal → resposta padrão do bot
  1     → Alerta → resposta empática + oferta de ajuda
  2+    → Crítico → TRANSBORDO imediato para humano
"""

# ------------------------------------------------------------------
# DICIONÁRIO DE SENTIMENTO
# ------------------------------------------------------------------
LEXICON_NEGATIVO = {
    # Palavras muito negativas (peso 2)
    'péssimo': 2, 'horroroso': 2, 'absurdo': 2, 'inadmissível': 2,
    'ódio': 2, 'lixo': 2, 'terrível': 2,
    # Palavras moderadamente negativas (peso 1)
    'quebrado': 1, 'atrasou': 1, 'ruim': 1, 'droga': 1,
    'problema': 1, 'errado': 1, 'decepcionado': 1, 'frustrado': 1,
    'demorou': 1, 'parou': 1,
}

LEXICON_POSITIVO = {
    'obrigado': 2, 'ótimo': 2, 'excelente': 2, 'perfeito': 2,
    'adorei': 2, 'parabéns': 2,
    'bom': 1, 'gostei': 1, 'ok': 1, 'certo': 1,
}

# ------------------------------------------------------------------
# FUNÇÕES DE ANÁLISE
# ------------------------------------------------------------------
def calcular_score_sentimento(mensagem: str) -> dict:
    """
    Retorna um dicionário com:
      - score_negativo : soma dos pesos negativos
      - score_positivo : soma dos pesos positivos
      - score_final    : positivo - negativo (negativo = mais irritado)
      - palavras_ruins : lista das palavras negativas encontradas
    """
    msg = mensagem.lower()
    
    score_neg = 0
    score_pos = 0
    palavras_ruins = []

    for palavra, peso in LEXICON_NEGATIVO.items():
        if palavra in msg:
            score_neg += peso
            palavras_ruins.append(palavra)

    for palavra, peso in LEXICON_POSITIVO.items():
        if palavra in msg:
            score_pos += peso

    return {
        "score_negativo":  score_neg,
        "score_positivo":  score_pos,
        "score_final":     score_pos - score_neg,
        "palavras_ruins":  palavras_ruins,
        "nivel_irritacao": score_neg,  # compatível com o exercício original
    }

def classificar_humor(score: dict) -> str:
    """Classifica o humor do cliente baseado no score."""
    nivel = score["nivel_irritacao"]
    if nivel == 0 and score["score_positivo"] > 0:
        return "POSITIVO 😊"
    if nivel == 0:
        return "NEUTRO 😐"
    if nivel == 1:
        return "INSATISFEITO 😟"
    return "IRRITADO 😡"


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("😊 Bot de Suporte com Análise de Sentimento")
print("   Transbordo automático ao detectar irritação severa")
print("   Digite 'sair' para encerrar")
print("=" * 55)
print("Bot: Olá! Como posso ajudar você hoje?\n")

turno = 0
while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Obrigado por entrar em contato. Até logo! 👋")
        break

    turno += 1

    # --- Análise de sentimento ---
    score = calcular_score_sentimento(entrada)
    humor = classificar_humor(score)

    # Indicador interno (em produção, isso seria log/métrica)
    print(f"[SISTEMA] Turno {turno} | Humor: {humor} | Score irritação: {score['nivel_irritacao']} | Palavras negativas: {score['palavras_ruins']}")

    # --- Lógica de gatilho ---
    if score["nivel_irritacao"] >= 2:
        print(
            "Bot: ⚠️  Detectei insatisfação severa na sua mensagem.\n"
            "     Protocolo de TRANSBORDO ativado.\n"
            "     Estou conectando você a um supervisor humano agora.\n"
            "     Por favor, aguarde — seu caso receberá prioridade máxima."
        )
        print(f"\n[SISTEMA] 🔴 TRANSBORDO disparado no turno {turno}. Agente humano notificado.")
        break

    elif score["nivel_irritacao"] == 1:
        print(
            "Bot: 😟 Percebo que você está tendo uma experiência difícil.\n"
            "     Lamento muito por isso! Vou dar prioridade ao seu caso.\n"
            "     Pode me dar mais detalhes para que eu resolva rapidamente?\n"
        )
    elif score["score_positivo"] > 0:
        print("Bot: 😊 Fico feliz em poder ajudar! Há mais alguma coisa que posso fazer?\n")
    else:
        print("Bot: Certo, entendi. Processando sua solicitação normalmente...\n")
