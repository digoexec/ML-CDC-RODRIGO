"""
=============================================================
EXERCÍCIO 5: Máquina de Estados Finita (FSM) com Menu Interativo
=============================================================

CONCEITO CENTRAL:
  Uma FSM (Finite State Machine) é um modelo matemático que define:
    - Um conjunto finito de ESTADOS possíveis
    - TRANSIÇÕES entre estados (disparadas por inputs)
    - Um estado INICIAL e estados FINAIS (opcionais)

  Em chatbots, isso garante que o usuário só possa fazer ações
  válidas para o contexto atual — prevenindo inputs inválidos e
  conversas incoerentes.

  DIAGRAMA DE ESTADOS deste bot:
  
  [MENU_PRINCIPAL]
       │  "1"        "2"
       ▼              ▼
  [SUPORTE]     [FINANCEIRO]
       │              │
       └──── "0" ─────┘
             │
       [MENU_PRINCIPAL]

ESTADOS DISPONÍVEIS:
  MENU_PRINCIPAL → hub de navegação
  SUPORTE        → módulo de atendimento técnico
  FINANCEIRO     → módulo de pagamentos e estornos

TRANSIÇÕES VÁLIDAS:
  MENU_PRINCIPAL + "1" → SUPORTE
  MENU_PRINCIPAL + "2" → FINANCEIRO
  SUPORTE ou FINANCEIRO + "0" → MENU_PRINCIPAL
  Qualquer estado + input inválido → permanece no mesmo estado
"""

# ------------------------------------------------------------------
# CONFIGURAÇÃO DOS ESTADOS E MENUS
# ------------------------------------------------------------------
MENUS = {
    "MENU_PRINCIPAL": {
        "descricao": "Menu Principal",
        "icone":     "🏠",
        "opcoes":    {"1": "SUPORTE", "2": "FINANCEIRO"},
        "texto_menu": "Opções disponíveis:\n  [1] Suporte Técnico\n  [2] Financeiro / Estornos",
    },
    "SUPORTE": {
        "descricao": "Suporte Técnico",
        "icone":     "🔧",
        "opcoes":    {"0": "MENU_PRINCIPAL"},
        "texto_menu": "Você está no módulo de SUPORTE.\n  [0] Voltar ao menu principal",
    },
    "FINANCEIRO": {
        "descricao": "Financeiro",
        "icone":     "💰",
        "opcoes":    {"0": "MENU_PRINCIPAL"},
        "texto_menu": "Você está no módulo FINANCEIRO.\n  [0] Voltar ao menu principal",
    },
}

# Respostas simuladas para inputs genéricos em cada módulo
RESPOSTAS_MODULO = {
    "SUPORTE": [
        "Verificando status do equipamento...",
        "Abrindo chamado técnico com prioridade normal.",
        "Conectando ao banco de dados de soluções conhecidas...",
    ],
    "FINANCEIRO": [
        "Consultando histórico de transações...",
        "Verificando estornos pendentes na sua conta.",
        "Acessando política de reembolso aplicável ao seu caso.",
    ],
}


# ------------------------------------------------------------------
# CLASSE DA FSM
# ------------------------------------------------------------------
class BotFSM:
    def __init__(self):
        self.estado_atual       = "MENU_PRINCIPAL"
        self.historico_estados  = ["MENU_PRINCIPAL"]  # rastrea o caminho percorrido
        self.contador_no_modulo = 0                   # interações dentro do módulo atual

    def _estado_info(self) -> dict:
        """Retorna a configuração do estado atual."""
        return MENUS[self.estado_atual]

    def transicionar(self, opcao: str) -> str:
        """
        Processa o input do usuário e:
          1. Verifica se a transição é válida
          2. Muda o estado (ou não, se inválido)
          3. Retorna a mensagem apropriada
        """
        info          = self._estado_info()
        opcoes_validas = info["opcoes"]

        # --- Transição válida ---
        if opcao in opcoes_validas:
            estado_destino     = opcoes_validas[opcao]
            self.estado_atual  = estado_destino
            self.historico_estados.append(estado_destino)
            self.contador_no_modulo = 0

            info_destino = MENUS[estado_destino]
            return (
                f"{info_destino['icone']} Entrando em [{info_destino['descricao'].upper()}].\n"
                f"{info_destino['texto_menu']}"
            )

        # --- Input inválido --- permanece no mesmo estado
        if opcao.isdigit():
            opcoes_str = ", ".join(opcoes_validas.keys())
            return (
                f"❌ Opção '{opcao}' não existe neste módulo.\n"
                f"   Opções válidas aqui: {opcoes_str}\n"
                f"{info['texto_menu']}"
            )

        # --- Input de texto livre dentro de um módulo ---
        if self.estado_atual in RESPOSTAS_MODULO:
            self.contador_no_modulo += 1
            respostas = RESPOSTAS_MODULO[self.estado_atual]
            resposta_idx = (self.contador_no_modulo - 1) % len(respostas)
            return (
                f"[{self.estado_atual}] {respostas[resposta_idx]}\n"
                f"   {info['texto_menu']}"
            )

        return f"Ainda processando no módulo {self.estado_atual}."

    def status(self) -> str:
        """Exibe o estado atual e o caminho percorrido."""
        caminho = " → ".join(self.historico_estados[-5:])  # últimos 5 estados
        return f"📍 Estado: [{self.estado_atual}] | Caminho: {caminho}"


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
bot = BotFSM()

print("=" * 55)
print("🤖 Bot FSM — Navegação por Estados Finitos")
print("   Digite 'status' para ver o estado atual")
print("   Digite 'sair' para encerrar")
print("=" * 55)
print(f"Bot: Bem-vindo! {MENUS['MENU_PRINCIPAL']['texto_menu']}\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Sessão encerrada. Obrigado! 👋")
        print(bot.status())
        break

    if entrada.lower() == 'status':
        print(f"Bot: {bot.status()}\n")
        continue

    resposta = bot.transicionar(entrada)
    print(f"Bot: {resposta}\n")
