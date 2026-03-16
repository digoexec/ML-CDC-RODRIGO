from collections import Counter
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

texto = "chatbot chatbot inteligência artificial chatbot aprendizado"

# TODO: transformar texto em lista de palavras
palavras = word_tokenize(texto)

# TODO: calcular frequência
frequencia = Counter(palavras)

print(frequencia)

# Resultado esperado: {'chatbot':3, 'inteligência':1, 'artificial':1, 'aprendizado':1}
