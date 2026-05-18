"""
=============================================================
EXERCÍCIO 1: Classificador de Intenções com TF-IDF + Naive Bayes
=============================================================

CONCEITO CENTRAL:
  Em vez de usar if/else para classificar mensagens, usamos IA clássica
  de Machine Learning. O pipeline funciona em 2 etapas:

  1. TF-IDF (Term Frequency - Inverse Document Frequency):
     Transforma texto em números (vetores). Palavras mais
     relevantes para um documento ganham peso maior.

  2. Naive Bayes (MultinomialNB):
     Algoritmo probabilístico que aprende os padrões dos vetores
     e prevê a qual categoria uma nova mensagem pertence.

FLUXO:
  CSV  →  rotulagem básica  →  TF-IDF  →  treina modelo  →  chat loop
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------------------------------------------------------
# 1. CARGA DOS DADOS
# ------------------------------------------------------------------
df = pd.read_csv('logs_ecommerce.csv')

print("📊 Amostra dos dados carregados:")
print(df[['id_usuario', 'mensagem_usuario']].head(3).to_string(index=False))
print()

# ------------------------------------------------------------------
# 2. ROTULAGEM (Label Engineering)
#    Criamos um classificador básico de regras para gerar os rótulos
#    de treino. Isso é chamado de "weak supervision".
# ------------------------------------------------------------------
def rotular_basico(msg):
    """Gera rótulos simples por palavras-chave — serve como ground truth inicial."""
    if 'Oi' in msg or 'Olá' in msg or 'bom dia' in msg.lower():
        return 'saudacao'
    if 'pedido' in msg or 'rastreio' in msg or 'endereço' in msg:
        return 'suporte'
    return 'reclamacao'

df['intencao'] = df['mensagem_usuario'].apply(rotular_basico)

print("🏷️  Distribuição das intenções no dataset de treino:")
print(df['intencao'].value_counts().to_string())
print()

# ------------------------------------------------------------------
# 3. VETORIZAÇÃO COM TF-IDF
#    Converte cada mensagem em um vetor numérico.
#    Ex: "meu pedido atrasou" → [0.0, 0.7, 0.3, ...]
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['mensagem_usuario'])  # matriz esparsa de features
y = df['intencao']                                     # rótulos alvo

print(f"📐 Vocabulário aprendido: {len(vectorizer.vocabulary_)} termos únicos")
print(f"📐 Forma da matriz de features: {X.shape} (linhas=mensagens, colunas=termos)\n")

# ------------------------------------------------------------------
# 4. TREINAMENTO DO MODELO
#    MultinomialNB aprende a probabilidade P(intenção | palavras)
# ------------------------------------------------------------------
modelo = MultinomialNB()
modelo.fit(X, y)

print("✅ Modelo treinado com sucesso!")
print("=" * 55)

# ------------------------------------------------------------------
# 5. CHAT LOOP — inferência em tempo real
# ------------------------------------------------------------------
print("\n🤖 Bot Classificador pronto! Digite sua mensagem (ou 'sair' para encerrar):\n")

while True:
    entrada = input("Você: ").strip()

    if not entrada:
        continue

    if entrada.lower() == 'sair':
        print("Bot: Até logo! 👋")
        break

    # Transforma a entrada nova com o mesmo vetorizador (sem re-treinar!)
    X_teste = vectorizer.transform([entrada])

    # Predição da classe + probabilidades de cada classe
    intencao = modelo.predict(X_teste)[0]
    probabilidades = modelo.predict_proba(X_teste)[0]
    confianca = max(probabilidades) * 100

    # Respostas personalizadas por intenção
    respostas = {
        'saudacao':   "Olá! Seja bem-vindo(a). Como posso te ajudar hoje?",
        'suporte':    "Entendido! Vou verificar as informações do seu pedido agora.",
        'reclamacao': "Lamentamos a experiência negativa. Vou escalar seu caso com prioridade.",
    }

    resposta = respostas.get(intencao, "Não entendi bem. Pode reformular?")
    print(f"Bot: [{intencao.upper()} | confiança: {confianca:.1f}%] {resposta}\n")
