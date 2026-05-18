"""
=============================================================
EXERCÍCIO 2: Gerenciamento de Contexto e Memória de Curto Prazo
=============================================================

CONCEITO CENTRAL:
  Chatbots sem estado sofrem de "amnésia": a cada resposta,
  esquecem tudo que foi dito antes. A solução é um Session State
  — um objeto Python que persiste durante toda a sessão de chat.

  Aqui implementamos isso com uma classe que guarda:
    - Nome do usuário
    - Último assunto discutido
    - Histórico completo de mensagens
    - Contador de interações

PADRÃO ARQUITETURAL:
  Classe com self.contexto (dict) → atualizada a cada turno
  → usada para personalizar todas as respostas seguintes.
"""

# ------------------------------------------------------------------
# CLASSE PRINCIPAL: Estado da Sessão
# ------------------------------------------------------------------
class ChatbotComMemoria:
    """
    Bot com memória de curto prazo baseada em Session State.
    O self.contexto é o "cérebro" que persiste entre turnos.
    """

    def __init__(self):
        # Dicionário de estado — a "memória" do bot
        self.contexto = {
            "nome":            None,   # Nome do usuário (coletado no 1º turno)
            "ultimo_assunto":  None,   # Último tópico detectado
            "num_interacoes":  0,      # Quantas mensagens já foram trocadas
            "historico":       [],     # Log completo da conversa
        }

    # ------------------------------------------------------------------
    # MÉTODO INTERNO: detecção de intenção simples por palavras-chave
    # ------------------------------------------------------------------
    def _detectar_intencao(self, mensagem: str) -> str:
        msg = mensagem.lower()
        if any(p in msg for p in ['comprar', 'preço', 'valor', 'custo', 'promoção']):
            return 'vendas'
        if any(p in msg for p in ['pedido', 'rastreio', 'entrega', 'prazo']):
            return 'suporte'
        if any(p in msg for p in ['problema', 'quebrado', 'errado', 'ruim']):
            return 'reclamacao'
        return 'geral'

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL: gera a resposta e atualiza o estado
    # ------------------------------------------------------------------
    def responder(self, mensagem: str) -> str:
        self.contexto["num_interacoes"] += 1
        self.contexto["historico"].append({"turno": self.contexto["num_interacoes"], "usuario": mensagem})

        nome = self.contexto["nome"]

        # --- TURNO 1: coleta o nome ---
        if not nome:
            self.contexto["nome"] = mensagem.strip().title()
            resposta = (
                f"Prazer, {self.contexto['nome']}! 😊 "
                f"Sou o assistente virtual da loja. Em que posso te ajudar hoje?"
            )

        # --- TURNOS SEGUINTES: responde com contexto ---
        else:
            intencao = self._detectar_intencao(mensagem)
            self.contexto["ultimo_assunto"] = intencao

            if intencao == 'vendas':
                resposta = (
                    f"Ótima escolha, {nome}! 🛍️ Nosso setor de vendas está com "
                    f"promoções incríveis essa semana. Quer ver as ofertas por categoria?"
                )
            elif intencao == 'suporte':
                resposta = (
                    f"Claro, {nome}! Vou verificar o status do seu pedido. "
                    f"Por favor, informe o número do pedido no formato #1234."
                )
            elif intencao == 'reclamacao':
                resposta = (
                    f"Lamentamos muito, {nome}. 😔 Sua satisfação é nossa prioridade. "
                    f"Vou registrar seu caso com prioridade máxima agora mesmo."
                )
            else:
                # Usa o histórico para enriquecer a resposta
                if self.contexto["ultimo_assunto"] and self.contexto["num_interacoes"] > 2:
                    resposta = (
                        f"Entendido, {nome}. Continuando nossa conversa sobre "
                        f"'{self.contexto['ultimo_assunto']}': posso te ajudar mais nesse assunto?"
                    )
                else:
                    resposta = f"Entendi sua mensagem, {nome}. Como posso ajudar mais?"

        self.contexto["historico"][-1]["bot"] = resposta
        return resposta

    def resumir_sessao(self):
        """Exibe um resumo do estado atual da sessão."""
        print("\n" + "=" * 45)
        print("📋 RESUMO DA SESSÃO:")
        print(f"  Usuário     : {self.contexto['nome']}")
        print(f"  Interações  : {self.contexto['num_interacoes']}")
        print(f"  Último tema : {self.contexto['ultimo_assunto']}")
        print("=" * 45)


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
bot = ChatbotComMemoria()

print("=" * 45)
print("🧠 Bot com Memória de Curto Prazo")
print("   Digite 'sair' para encerrar")
print("=" * 45)
print("Bot: Olá! Qual é o seu nome?\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        bot.resumir_sessao()
        print("Bot: Até logo! Foi um prazer te atender. 👋")
        break

    resposta = bot.responder(entrada)
    print(f"Bot: {resposta}\n")
