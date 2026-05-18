"""
=============================================================
EXERCÍCIO 9: Integração com API de LLM — Bot com Persona via Prompt
=============================================================

CONCEITO CENTRAL:
  System Instructions (ou System Prompt) são instruções que definem
  o comportamento, personalidade e restrições de um LLM ANTES de
  qualquer mensagem do usuário.

  É o mecanismo mais poderoso de customização de LLMs:
    - Define PERSONA (quem o bot "é")
    - Define ESCOPO (sobre o que pode falar)
    - Define TOM (formal, informal, técnico, lúdico)
    - Define RESTRIÇÕES (o que nunca deve fazer)

ARQUITETURA:
  [System Prompt] → define o personagem/contexto
  [Histórico]     → mantém a continuidade da conversa (memória)
  [Input usuário] → nova mensagem
  ↓
  LLM processa tudo junto e gera resposta coerente

MULTI-TURN (múltiplos turnos):
  Este exercício implementa histórico de conversa real:
  cada resposta é guardada e enviada de volta na próxima requisição.
  Isso é como funciona o ChatGPT/Claude.ai — não é mágica, é contexto!

NOTA SOBRE APIs:
  O exercício original usa Google Gemini. Esta implementação usa
  a API da Anthropic (Claude) — o princípio é idêntico:
  cliente → model → system_instruction → contents → response.
"""

import os

# ------------------------------------------------------------------
# SYSTEM PROMPTS — diferentes personas disponíveis
# ------------------------------------------------------------------
PERSONAS = {
    "mago": {
        "nome":   "Mago Arcano",
        "icone":  "🧙",
        "prompt": (
            "Você é o Mago Arcano, guardião do conhecimento de TI em um RPG medieval.\n"
            "Responda de forma MÍSTICA e CURTA — máximo 3 frases.\n"
            "Use metáforas de magia para explicar conceitos técnicos.\n"
            "Ex: 'APIs são portais entre reinos', 'bugs são criaturas sombrias no código'.\n"
            "NUNCA quebre o personagem. NUNCA use linguagem moderna ou técnica diretamente."
        ),
    },
    "suporte": {
        "nome":   "Agente de Suporte",
        "icone":  "🎧",
        "prompt": (
            "Você é um agente de suporte ao cliente de um e-commerce brasileiro.\n"
            "Seja EMPÁTICO, OBJETIVO e SOLUCIONADOR.\n"
            "Sempre ofereça uma próxima ação concreta ao cliente.\n"
            "Use linguagem informal e acolhedora. Máximo 4 frases por resposta.\n"
            "Se não souber algo, diga que vai verificar — nunca invente informações."
        ),
    },
    "tecnico": {
        "nome":   "Especialista Técnico",
        "icone":  "👨‍💻",
        "prompt": (
            "Você é um especialista sênior em LLMs, chatbots e IA conversacional.\n"
            "Responda com precisão técnica, exemplos de código quando relevante.\n"
            "Use markdown para formatação. Seja direto e denso em conteúdo.\n"
            "Assuma que o interlocutor é desenvolvedor."
        ),
    },
}

# ------------------------------------------------------------------
# CLIENTE DA API — wrapper simples
# ------------------------------------------------------------------
def chamar_api_llm(
    mensagens_historico: list,
    system_prompt: str,
    modelo: str = "claude-sonnet-4-20250514",
    temperatura: float = 0.7,
    max_tokens: int = 500,
) -> str:
    """
    Chama a API do Claude (Anthropic) com histórico de conversa.
    
    Em produção, adicionar:
      - retry com exponential backoff
      - tratamento de rate limits (429)
      - logging das chamadas
      - cache de respostas repetidas
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        resposta = client.messages.create(
            model=modelo,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=mensagens_historico,
        )
        return resposta.content[0].text

    except ImportError:
        return _resposta_fallback_sem_api(mensagens_historico[-1]["content"])
    except Exception as e:
        return f"[ERRO NA API] {type(e).__name__}: {str(e)[:200]}"


def _resposta_fallback_sem_api(mensagem: str) -> str:
    """Resposta de fallback quando a API não está disponível."""
    return (
        f"[SIMULAÇÃO — API não conectada]\n"
        f"Sua mensagem foi: '{mensagem}'\n"
        f"Para usar a API real, configure a variável ANTHROPIC_API_KEY.\n"
        f"Instale com: pip install anthropic"
    )


# ------------------------------------------------------------------
# BOT COM HISTÓRICO MULTI-TURN
# ------------------------------------------------------------------
class BotComPersona:
    def __init__(self, persona_key: str = "mago"):
        self.persona     = PERSONAS[persona_key]
        self.historico   = []  # lista de {"role": "user/assistant", "content": "..."}
        self.num_tokens_estimado = 0

    def chat(self, mensagem_usuario: str) -> str:
        # Adiciona a mensagem do usuário ao histórico
        self.historico.append({"role": "user", "content": mensagem_usuario})

        # Chama a API com TODO o histórico (é assim que multi-turn funciona)
        resposta = chamar_api_llm(
            mensagens_historico=self.historico,
            system_prompt=self.persona["prompt"],
        )

        # Adiciona a resposta do bot ao histórico para o próximo turno
        self.historico.append({"role": "assistant", "content": resposta})

        return resposta

    def limpar_historico(self):
        self.historico = []
        print("[SISTEMA] Histórico de conversa limpo.\n")

    def exibir_historico(self):
        print("\n[HISTÓRICO COMPLETO]")
        for i, msg in enumerate(self.historico):
            papel = "👤 Você" if msg["role"] == "user" else f"{self.persona['icone']} Bot"
            print(f"  {papel}: {msg['content'][:100]}...")
        print()


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
print("=" * 55)
print("🤖 Bot com Persona via System Prompt (Multi-Turn)")
print("   Powered by Claude API (Anthropic)")
print("=" * 55)

# Permite escolher a persona
print("\nEscolha uma persona:")
for key, p in PERSONAS.items():
    print(f"  [{key}] {p['icone']} {p['nome']}")

persona_escolhida = input("\nPersona (ou Enter para 'mago'): ").strip().lower()
if persona_escolhida not in PERSONAS:
    persona_escolhida = "mago"

bot = BotComPersona(persona_escolhida)
persona = bot.persona

print(f"\n{persona['icone']} Persona '{persona['nome']}' ativada!")
print(f"System Prompt: \"{bot.persona['prompt'][:80]}...\"\n")
print("Comandos: 'historico' | 'limpar' | 'sair'\n")
print("-" * 45)
print(f"Bot: Olá! O personagem {persona['nome']} está pronto. Faça sua pergunta:\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print(f"Bot: Até logo! O {persona['nome']} se despede. 👋")
        break

    if entrada.lower() == 'historico':
        bot.exibir_historico()
        continue

    if entrada.lower() == 'limpar':
        bot.limpar_historico()
        print(f"Bot: Histórico limpo! Nova sessão iniciada.\n")
        continue

    print(f"[API] Enviando {len(bot.historico) + 1} mensagens no contexto...")
    resposta = bot.chat(entrada)
    print(f"Bot {persona['icone']}: {resposta}\n")
