# 🏥 Chatbot Inteligente - Sistema de Predição de Risco Clínico

## Visão Geral

Um **assistente de saúde conversacional** totalmente integrado ao Sistema de Predição de Risco Clínico. O chatbot utiliza inteligência artificial para fornecer respostas contextualizadas sobre biomedicina, interpretação de risco e recomendações de saúde.

## ✨ Características Principais

### 1. **Interface Elegante e Responsiva**
- Botão flutuante com animações suaves
- Janela de chat moderna com design dark mode
- Totalmente responsivo para mobile e desktop
- Animações de entrada e transição profissionais

### 2. **IA Conversacional Inteligente**
- Reconhecimento de intenção do usuário
- Respostas contextualizadas baseadas em base de conhecimento
- Suporte a perguntas sobre:
  - Funcionamento do modelo Random Forest
  - Interpretação de níveis de risco
  - Explicação de parâmetros biomédicos
  - Recomendações de saúde
  - Perguntas frequentes

### 3. **Sugestões Rápidas**
Botões de atalho para perguntas comuns:
- 📊 Como funciona o modelo?
- ⚠️ Entender meu risco
- 💡 Recomendações
- 📋 Sobre os parâmetros

### 4. **Histórico de Conversa**
- Mantém histórico completo da conversa
- Timestamps para cada mensagem
- Scroll automático para novas mensagens
- Indicador de digitação realista

### 5. **Integração com Sistema Principal**
- Captura dados do paciente após cálculo de risco
- Fornece recomendações personalizadas
- Contextualiza respostas com dados do usuário

## 🚀 Como Usar

### Abrir o Chatbot
1. Clique no botão flutuante azul no canto inferior direito
2. A janela de chat se abrirá com uma saudação amigável

### Interagir com o Chatbot
- **Digite sua pergunta** no campo de entrada
- **Pressione Enter** ou clique no botão de envio
- **Clique nas sugestões rápidas** para perguntas pré-definidas
- **Feche o chat** clicando no botão X

## 📋 Base de Conhecimento

### Tópicos Cobertos

#### 1. **Modelo de Machine Learning**
- Explicação do Random Forest
- Acurácia e métricas de desempenho
- Parâmetros utilizados

#### 2. **Parâmetros Biomédicos**
- **Idade**: Fator de risco etário
- **Glicose**: Classificação de diabetes
- **Pressão Arterial**: Classificação de hipertensão
- **IMC**: Classificação de peso
- **Colesterol**: Classificação de risco cardiovascular

#### 3. **Recomendações de Saúde**
- 🥗 Alimentação saudável
- 🏃 Atividade física
- 🧘 Gerenciamento de estresse
- 😴 Qualidade do sono
- 📊 Monitoramento regular

#### 4. **Interpretação de Risco**
- **Baixo Risco**: Padrão saudável
- **Risco Médio**: Atenção recomendada
- **Risco Alto**: Intervenção urgente

## 🎨 Design e Estilo

### Cores
- Fundo: #111827 (Dark Blue-Gray)
- Primária: #3b82f6 (Blue)
- Sucesso: #10b981 (Green)
- Aviso: #f59e0b (Amber)
- Perigo: #ef4444 (Red)

### Tipografia
- Font: IBM Plex Sans (corpo)
- Font: IBM Plex Mono (dados técnicos)

### Animações
- Entrada suave (slide-in)
- Fade-in para mensagens
- Pulse para badge de notificação
- Bounce para indicador de digitação

## 📱 Responsividade

O chatbot se adapta automaticamente:
- **Desktop**: Janela 380px de largura
- **Tablet**: Ajuste de tamanho
- **Mobile**: Ocupa até 100vw - 40px com altura máxima de 70vh

## 🔧 Arquivos

### `chatbot.js`
- Lógica principal do chatbot
- Classe `BiomedicinaChatbot`
- Base de conhecimento
- Geração de respostas
- Gerenciamento de eventos

### `chatbot.css`
- Estilos da interface
- Animações
- Responsividade
- Temas (dark/light)

### `sistema_risco_clinico.html`
- Integração do chatbot
- Links aos arquivos CSS e JS
- Função de atualização com dados do paciente

## 💡 Exemplos de Uso

### Pergunta 1: Sobre o Modelo
**Usuário**: "Como o modelo funciona?"
**Chatbot**: Explica o Random Forest e seus parâmetros

### Pergunta 2: Sobre Parâmetros
**Usuário**: "O que é glicose?"
**Chatbot**: Explica a importância da glicose e faixas de normalidade

### Pergunta 3: Recomendações
**Usuário**: "Como reduzir meu risco?"
**Chatbot**: Lista recomendações de estilo de vida

## 🎯 Funcionalidades Avançadas

### Reconhecimento de Intenção
O chatbot identifica automaticamente:
- Saudações
- Perguntas sobre modelo
- Perguntas sobre acurácia
- Perguntas sobre risco
- Perguntas sobre parâmetros
- Perguntas sobre recomendações

### Formatação de Mensagens
- **Negrito**: `**texto**`
- **Itálico**: `__texto__`
- **Quebras de linha**: Suportadas automaticamente

### Indicador de Digitação
Mostra animação realista enquanto o chatbot "digita" a resposta

## 🔐 Segurança

- Nenhum dado pessoal é enviado para servidores externos
- Processamento 100% no cliente (navegador)
- Histórico de conversa armazenado apenas na sessão atual

## 📊 Métricas de Desempenho

- Tempo de resposta: < 1 segundo
- Tamanho do arquivo JS: ~17KB
- Tamanho do arquivo CSS: ~16KB
- Sem dependências externas

## 🚀 Melhorias Futuras

- [ ] Integração com API de IA para respostas mais naturais
- [ ] Suporte a múltiplos idiomas
- [ ] Persistência de histórico
- [ ] Análise de sentimento
- [ ] Recomendações personalizadas baseadas em ML
- [ ] Integração com sistema de agendamento
- [ ] Feedback do usuário

## 📞 Suporte

Para dúvidas ou sugestões sobre o chatbot, consulte a documentação do projeto ou entre em contato com a equipe de desenvolvimento.

---

**Desenvolvido com ❤️ para a Clínica de Biomedicina**
**Versão 1.0 | Maio 2026**
