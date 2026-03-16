# Exercício: criar um contador de palavras, aqui você vai precisar construir e completar o código
import nltk

nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

texto = "machine learning é importante para inteligência artificial"

# TODO: dividir o texto em palavras
palavras = word_tokenize(texto)

# TODO: contar quantas palavras existem
quantidade = len(palavras)

print("Número de palavras:", quantidade)

#Resultado esperado - Número de palavras: 7
