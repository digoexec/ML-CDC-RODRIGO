"""
=============================================================
EXERCÍCIO 6: FAQ Inteligente com Similaridade de Cosseno
=============================================================

CONCEITO CENTRAL:
  Similaridade de Cosseno mede o ângulo entre dois vetores no espaço
  matemático. Vetores com ângulo 0° têm similaridade 1.0 (idênticos);
  ângulo 90° = 0.0 (completamente diferentes).

  Isso nos permite comparar SIGNIFICADO sem exigir palavras exatas.

  EXEMPLO:
    FAQ:    "Como rastrear meu pedido"
    Usuário: "quero ver onde está minha encomenda"
    → palavras diferentes, mas similaridade alta!

PIPELINE:
  1. CountVectorizer transforma texto em vetores de contagem de palavras
  2. cosine_similarity calcula a distância entre o vetor da pergunta
     do usuário e cada vetor da base de FAQ
  3. O índice com maior similaridade vira a resposta

POR QUE COSSENO E NÃO DISTÂNCIA EUCLIDIANA?
  Distância euclidiana favorece textos mais longos (mais palavras = 
  vetor maior). O cosseno normaliza isso — só o ângulo importa.

THRESHOLD (limiar):
  Se a similaridade for < 0.2, consideramos "sem resposta relevante".
  Isso evita respostas erradas com alta confiança.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------------------------------
# BASE DE FAQ — em produção viria de um banco de dados
# ------------------------------------------------------------------
FAQ_BASE = [
    {
        "pergunta": "Como posso rastrear meu pedido",
        "resposta":  "Acesse 'Meus Pedidos' no app, clique no pedido e veja o código de rastreio dos Correios.",
        "tags":      ["rastreio", "pedido", "entrega", "encomenda"],
    },
    {
        "pergunta": "Quais as formas de pagamento aceitas",
        "resposta":  "Aceitamos cartão de crédito, débito, Pix, boleto bancário e carteiras digitais (Mercado Pago, PicPay).",
        "tags":      ["pagamento", "cartão", "pix", "boleto"],
    },
    {
        "pergunta": "Como funciona a política de troca",
        "resposta":  "Trocas são aceitas em até 30 dias após o recebimento. O produto deve estar sem uso e na embalagem original.",
        "tags":      ["troca", "devolução", "reembolso", "prazo"],
    },
    {
        "pergunta": "Qual o prazo de entrega",
        "resposta":  "O prazo varia por região: capitais 3-5 dias úteis, interior 5-10 dias úteis. Consulte o CEP na hora da compra.",
        "tags":      ["prazo", "entrega", "dias", "frete"],
    },
    {
        "pergunta": "Como cancelar um pedido",
        "resposta":  "Pedidos podem ser cancelados em até 24h após a compra pelo app. Após esse prazo, entre em contato com o suporte.",
        "tags":      ["cancelar", "cancelamento", "desistir"],
    },
]

# Extrai apenas as perguntas para vetorização
PERGUNTAS_FAQ = [item["pergunta"] for item in FAQ_BASE]


# ------------------------------------------------------------------
# FUNÇÃO DE BUSCA POR SIMILARIDADE
# ------------------------------------------------------------------
def buscar_faq(pergunta_usuario: str, threshold: float = 0.2) -> dict:
    """
    Compara a pergunta do usuário com toda a base FAQ via cosseno.
    Retorna o item mais próximo se acima do threshold, ou None.
    """
    # Junta FAQ + pergunta do usuário em uma lista só para vetorizar juntas
    todas_frases = PERGUNTAS_FAQ + [pergunta_usuario]

    # CountVectorizer: cria matriz de termos (linhas=frases, colunas=palavras)
    vectorizer = CountVectorizer().fit_transform(todas_frases)
    vetores    = vectorizer.toarray()

    # Calcula similaridade entre a última linha (usuário) e todas as do FAQ
    similitudes   = cosine_similarity(vetores[-1:], vetores[:-1])
    melhor_indice = similitudes.argmax()
    melhor_score  = similitudes[0][melhor_indice]

    return {
        "encontrou":     melhor_score >= threshold,
        "score":         melhor_score,
        "indice":        melhor_indice,
        "item_faq":      FAQ_BASE[melhor_indice] if melhor_score >= threshold else None,
        "todos_scores":  [(FAQ_BASE[i]["pergunta"], round(similitudes[0][i], 3))
                          for i in range(len(FAQ_BASE))],
    }


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("📚 Bot FAQ — Busca por Similaridade de Cosseno")
print("   Faça perguntas com suas próprias palavras!")
print("   Digite 'debug' para ver todos os scores de similaridade")
print("   Digite 'sair' para encerrar")
print("=" * 55)
print("Bot: Olá! Tenho respostas sobre rastreio, pagamento, trocas,")
print("     prazos e cancelamentos. Qual é a sua dúvida?\n")

modo_debug = False
while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Até logo! 👋")
        break

    if entrada.lower() == 'debug':
        modo_debug = not modo_debug
        print(f"Bot: Modo debug {'ATIVADO 🔍' if modo_debug else 'desativado'}.\n")
        continue

    # --- Busca na FAQ ---
    resultado = buscar_faq(entrada)

    if modo_debug:
        print("  [DEBUG] Scores de similaridade:")
        for pergunta, score in sorted(resultado["todos_scores"], key=lambda x: -x[1]):
            barra = "█" * int(score * 20)
            print(f"    {score:.3f} {barra} → {pergunta}")
        print()

    if resultado["encontrou"]:
        item = resultado["item_faq"]
        confianca = resultado["score"] * 100
        print(f"Bot: 🎯 [{confianca:.0f}% de relevância]\n"
              f"     Pergunta mais próxima no FAQ: '{item['pergunta']}'\n"
              f"     Resposta: {item['resposta']}\n")
    else:
        score = resultado["score"] * 100
        print(f"Bot: ❓ Não encontrei uma resposta relevante (maior match: {score:.0f}%).\n"
              f"     Tente reformular ou use palavras como: rastreio, pagamento, troca, prazo, cancelamento.\n")
