/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CHATBOT INTELIGENTE PARA CLÍNICA DE BIOMEDICINA
 * Sistema de IA conversacional para assistência clínica e interpretação de risco
 * ═══════════════════════════════════════════════════════════════════════════
 */

class BiomedicinaChatbot {
  constructor() {
    this.conversationHistory = [];
    this.currentRiskLevel = null;
    this.patientData = null;
    this.isOpen = false;
    this.isLoading = false;
    
    // Base de conhecimento para respostas contextualizadas
    this.knowledgeBase = {
      greetings: [
        "Olá! 👋 Bem-vindo ao assistente de saúde da clínica. Como posso ajudá-lo hoje?",
        "Oi! Sou o assistente de risco clínico. Estou aqui para esclarecer dúvidas sobre sua avaliação de saúde.",
        "Bem-vindo! 🏥 Posso ajudar com informações sobre risco clínico, parâmetros biomédicos e recomendações de saúde."
      ],
      
      riskExplanations: {
        low: "Seu perfil indica **BAIXO RISCO** clínico. Seus parâmetros biomédicos estão dentro dos limites saudáveis. Recomendações: manter hábitos saudáveis, exercícios regulares e monitoramento anual.",
        medium: "Seu perfil indica **RISCO MÉDIO**. Alguns parâmetros requerem atenção. Recomendações: consulta médica semestral, revisar alimentação, aumentar atividade física e monitorar pressão/glicose.",
        high: "Seu perfil indica **RISCO ALTO**. Intervenção médica é recomendada. Recomendações: procurar atendimento médico imediato, investigação clínica detalhada e acompanhamento especializado."
      },
      
      parameterInfo: {
        idade: "**Idade**: Fator importante no risco cardiovascular. Pessoas com 60+ anos têm risco aumentado. Mantenha acompanhamento regular.",
        glicose: "**Glicose**: Normal < 100 mg/dL | Pré-diabético 100-125 | Diabético ≥ 126. Controle através de dieta e exercícios.",
        pressao: "**Pressão Arterial**: Normal < 130 | Hipertensão Grau 1: 130-139 | Grau 2: ≥ 140 mmHg. Reduza sal e estresse.",
        imc: "**IMC (Índice de Massa Corporal)**: Normal < 25 | Sobrepeso 25-29.9 | Obesidade ≥ 30. Afeta risco cardiovascular.",
        colesterol: "**Colesterol Total**: Desejável < 200 | Limítrofe 200-239 | Alto ≥ 240 mg/dL. Dieta baixa em gordura saturada é essencial."
      },
      
      recommendations: {
        diet: "🥗 **Alimentação Saudável**: Reduza sal, açúcar e gordura saturada. Aumente fibras, frutas, verduras e proteínas magras.",
        exercise: "🏃 **Atividade Física**: 150 min/semana de exercício moderado. Caminhada, natação ou ciclismo são excelentes opções.",
        stress: "🧘 **Gerenciamento de Estresse**: Meditação, yoga ou técnicas de respiração reduzem pressão arterial.",
        sleep: "😴 **Sono**: 7-8 horas diárias melhoram metabolismo e reduzem risco cardiovascular.",
        monitoring: "📊 **Monitoramento**: Acompanhe regularmente seus parâmetros. Consulte médico conforme recomendação."
      },
      
      faq: {
        "Como o modelo funciona?": "O sistema usa **Random Forest**, um algoritmo de machine learning que analisa 5 parâmetros biomédicos (idade, glicose, pressão, IMC, colesterol) para calcular seu nível de risco clínico com 93.5% de acurácia.",
        
        "Qual é a acurácia do modelo?": "O modelo alcança **93.5% de acurácia** no conjunto de teste, com F1-Score de 0.9362 e AUC médio de 0.9915. Isso significa alta confiabilidade nas predições.",
        
        "Posso confiar neste resultado?": "Este é um **sistema de apoio clínico**, não substitui diagnóstico médico. Use como ferramenta de triagem e sempre consulte um profissional de saúde.",
        
        "Com que frequência devo fazer a avaliação?": "Recomenda-se avaliação **anual para baixo risco**, **semestral para risco médio** e **trimestral ou conforme orientação médica para risco alto**.",
        
        "O que fazer se meu risco for alto?": "Procure atendimento médico imediato. Solicite investigação clínica detalhada, exames complementares e acompanhamento especializado.",
        
        "Como reduzir meu risco?": "Através de **mudanças no estilo de vida**: dieta saudável, exercícios regulares, controle de peso, gerenciamento de estresse e sono adequado."
      }
    };
    
    this.init();
  }
  
  /**
   * Inicializa o chatbot e cria elementos DOM
   */
  init() {
    this.createChatbotUI();
    this.attachEventListeners();
  }
  
  /**
   * Cria a interface do chatbot
   */
  createChatbotUI() {
    // Container principal
    const chatbotContainer = document.createElement('div');
    chatbotContainer.id = 'biomedicina-chatbot';
    chatbotContainer.innerHTML = `
      <div class="chatbot-widget">
        <!-- Botão flutuante -->
        <button class="chatbot-toggle" title="Abrir assistente de saúde">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
          </svg>
          <span class="chatbot-badge">1</span>
        </button>
        
        <!-- Janela de chat -->
        <div class="chatbot-window">
          <!-- Header -->
          <div class="chatbot-header">
            <div class="chatbot-header-content">
              <h3>Assistente de Saúde 🏥</h3>
              <p>Dúvidas sobre sua avaliação?</p>
            </div>
            <button class="chatbot-close" title="Fechar chat">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
              </svg>
            </button>
          </div>
          
          <!-- Mensagens -->
          <div class="chatbot-messages" id="chatbot-messages">
            <div class="chat-message bot-message">
              <div class="message-content">
                Olá! 👋 Bem-vindo ao assistente de saúde da clínica. Como posso ajudá-lo hoje?
              </div>
              <div class="message-time">agora</div>
            </div>
          </div>
          
          <!-- Sugestões rápidas -->
          <div class="chatbot-suggestions" id="chatbot-suggestions">
            <button class="suggestion-btn" data-action="explain-model">
              📊 Como funciona o modelo?
            </button>
            <button class="suggestion-btn" data-action="understand-risk">
              ⚠️ Entender meu risco
            </button>
            <button class="suggestion-btn" data-action="recommendations">
              💡 Recomendações
            </button>
            <button class="suggestion-btn" data-action="parameters">
              📋 Sobre os parâmetros
            </button>
          </div>
          
          <!-- Input -->
          <div class="chatbot-input-area">
            <input 
              type="text" 
              id="chatbot-input" 
              class="chatbot-input" 
              placeholder="Digite sua pergunta..."
              autocomplete="off"
            >
            <button class="chatbot-send" title="Enviar mensagem">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M16.6915026,12.4744748 L3.50612381,13.2599618 C3.19218622,13.2599618 3.03521743,13.4170592 3.03521743,13.5741566 L1.15159189,20.0151496 C0.8376543,20.8006365 0.99,21.89 1.77946707,22.52 C2.41,22.99 3.50612381,23.1 4.13399899,22.8429026 L21.714504,14.0454487 C22.6563168,13.5741566 23.1272231,12.6315722 22.9702544,11.6889879 L4.13399899,1.16151496 C3.34915502,0.9 2.40734225,0.9 1.77946707,1.4429026 C0.994623095,2.0766 0.837654326,3.1659 1.15159189,3.9513869 L3.03521743,10.3923799 C3.03521743,10.5494773 3.19218622,10.7065747 3.50612381,10.7065747 L16.6915026,11.4920616 C16.6915026,11.4920616 17.1624089,11.4920616 17.1624089,12.0349652 C17.1624089,12.4744748 16.6915026,12.4744748 16.6915026,12.4744748 Z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(chatbotContainer);
  }
  
  /**
   * Anexa event listeners aos elementos
   */
  attachEventListeners() {
    const toggle = document.querySelector('.chatbot-toggle');
    const close = document.querySelector('.chatbot-close');
    const sendBtn = document.querySelector('.chatbot-send');
    const input = document.getElementById('chatbot-input');
    const suggestions = document.querySelectorAll('.suggestion-btn');
    
    toggle.addEventListener('click', () => this.toggleChat());
    close.addEventListener('click', () => this.closeChat());
    sendBtn.addEventListener('click', () => this.sendMessage());
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    
    suggestions.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const action = e.target.closest('.suggestion-btn').dataset.action;
        this.handleSuggestion(action);
      });
    });
  }
  
  /**
   * Alterna visibilidade do chat
   */
  toggleChat() {
    const window = document.querySelector('.chatbot-window');
    this.isOpen = !this.isOpen;
    
    if (this.isOpen) {
      window.classList.add('open');
      document.getElementById('chatbot-input').focus();
    } else {
      window.classList.remove('open');
    }
  }
  
  /**
   * Fecha o chat
   */
  closeChat() {
    const window = document.querySelector('.chatbot-window');
    this.isOpen = false;
    window.classList.remove('open');
  }
  
  /**
   * Envia mensagem do usuário
   */
  sendMessage() {
    const input = document.getElementById('chatbot-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Adiciona mensagem do usuário
    this.addMessage(message, 'user');
    input.value = '';
    
    // Simula digitação e gera resposta
    this.showTypingIndicator();
    
    setTimeout(() => {
      const response = this.generateResponse(message);
      this.removeTypingIndicator();
      this.addMessage(response, 'bot');
    }, 800 + Math.random() * 400);
  }
  
  /**
   * Adiciona mensagem ao histórico
   */
  addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    // Parse markdown simples
    const formattedText = this.formatMessage(text);
    
    messageDiv.innerHTML = `
      <div class="message-content">${formattedText}</div>
      <div class="message-time">${this.getTimeString()}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    this.conversationHistory.push({ sender, text });
  }
  
  /**
   * Formata mensagem com markdown simples
   */
  formatMessage(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.*?)__/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }
  
  /**
   * Mostra indicador de digitação
   */
  showTypingIndicator() {
    const messagesContainer = document.getElementById('chatbot-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
      <div class="message-content">
        <span></span><span></span><span></span>
      </div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
  
  /**
   * Remove indicador de digitação
   */
  removeTypingIndicator() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();
  }
  
  /**
   * Gera resposta baseada na entrada do usuário
   */
  generateResponse(userMessage) {
    const msg = userMessage.toLowerCase();
    
    // Verificar saudações
    if (msg.match(/oi|olá|opa|e aí|hey|opa/i)) {
      return this.knowledgeBase.greetings[Math.floor(Math.random() * this.knowledgeBase.greetings.length)];
    }
    
    // Verificar perguntas sobre modelo
    if (msg.match(/como.*funciona|modelo|algoritmo|machine learning|rf|random forest/i)) {
      return this.knowledgeBase.faq["Como o modelo funciona?"];
    }
    
    // Verificar perguntas sobre acurácia
    if (msg.match(/acurácia|precisão|confiança|confiável|f1|auc/i)) {
      return this.knowledgeBase.faq["Qual é a acurácia do modelo?"];
    }
    
    // Verificar perguntas sobre risco
    if (msg.match(/risco|alto|médio|baixo|nivel|nível|classificação/i)) {
      if (this.currentRiskLevel) {
        return this.knowledgeBase.riskExplanations[this.currentRiskLevel];
      }
      return "Para entender seu nível de risco, use o formulário acima para inserir seus parâmetros biomédicos e clique em 'Calcular Risco'.";
    }
    
    // Verificar perguntas sobre parâmetros específicos
    if (msg.match(/idade|glicose|pressão|imc|colesterol|peso|altura/i)) {
      for (const [param, info] of Object.entries(this.knowledgeBase.parameterInfo)) {
        if (msg.includes(param)) {
          return info;
        }
      }
      return "Posso explicar sobre: **Idade**, **Glicose**, **Pressão Arterial**, **IMC** ou **Colesterol**. Qual você gostaria de saber?";
    }
    
    // Verificar perguntas sobre recomendações
    if (msg.match(/recomendação|dica|como.*reduzir|melhorar|saúde|dieta|exercício|atividade/i)) {
      const recommendations = Object.values(this.knowledgeBase.recommendations);
      return recommendations.join('\n\n');
    }
    
    // Verificar perguntas frequentes
    for (const [question, answer] of Object.entries(this.knowledgeBase.faq)) {
      if (msg.match(new RegExp(question.split(' ').slice(0, 2).join('|'), 'i'))) {
        return answer;
      }
    }
    
    // Resposta padrão amigável
    const defaultResponses = [
      "Ótima pergunta! 🤔 Posso ajudar com informações sobre: risco clínico, parâmetros biomédicos, recomendações de saúde ou como funciona o modelo.",
      "Entendi sua dúvida! Você pode me perguntar sobre: **modelo**, **risco**, **parâmetros**, **recomendações** ou **acurácia**.",
      "Não tenho informação específica sobre isso, mas posso ajudar com: explicação do modelo, interpretação de risco, dicas de saúde ou informações sobre os parâmetros.",
      "Boa pergunta! 💡 Clique em uma das sugestões abaixo ou me faça outra pergunta sobre saúde e risco clínico."
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  }
  
  /**
   * Trata cliques em sugestões rápidas
   */
  handleSuggestion(action) {
    const suggestions = {
      'explain-model': 'Como o modelo Random Forest funciona?',
      'understand-risk': 'Como entender meu nível de risco?',
      'recommendations': 'Quais são as principais recomendações para reduzir risco?',
      'parameters': 'Explique os parâmetros biomédicos utilizados'
    };
    
    const message = suggestions[action];
    if (message) {
      this.addMessage(message, 'user');
      this.showTypingIndicator();
      
      setTimeout(() => {
        const response = this.generateResponse(message);
        this.removeTypingIndicator();
        this.addMessage(response, 'bot');
      }, 800 + Math.random() * 400);
    }
  }
  
  /**
   * Atualiza dados do paciente (chamado pelo sistema principal)
   */
  updatePatientData(idade, glicose, pressao, imc, colesterol, riskLevel) {
    this.patientData = { idade, glicose, pressao, imc, colesterol };
    this.currentRiskLevel = riskLevel;
  }
  
  /**
   * Retorna string de hora formatada
   */
  getTimeString() {
    const now = new Date();
    return now.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
  }
}

// Inicializa o chatbot quando o DOM está pronto
document.addEventListener('DOMContentLoaded', () => {
  window.biomedicinaChatbot = new BiomedicinaChatbot();
});
