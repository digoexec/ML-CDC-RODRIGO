import os
from google import genai
from google.genai import types

# Configuração da API Key (deve ser definida no ambiente)
# No sandbox, simularemos a execução para o teste solicitado.
api_key = os.environ.get("GEMINI_API_KEY", "SUA_API_KEY")

def iniciar_bot():
    print("Gemini Bot Persona: Iniciando conexão com o ecossistema Google AI. Faça sua pergunta técnica:")
    
    # Instanciando o cliente
    client = genai.Client(api_key=api_key)
    
    # Configurações do modelo com System Instruction (Persona Mística)
    config_ia = types.GenerateContentConfig(
        system_instruction="Você é o mestre dos magos de um RPG de TI. Fale de forma mística, curta e use metáforas de fantasia para conceitos de tecnologia.",
        temperature=0.7
    )

    while True:
        try:
            entrada = input("Você: ")
            if entrada.lower() in ['sair', 'exit', 'quit']:
                print("Bot: Que os deuses do silício guiem seus pacotes... Até breve, viajante.")
                break
            
            # Gerando conteúdo
            resposta = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=entrada,
                config=config_ia
            )
            
            print(f"Bot: {resposta.text}\n")
            
        except Exception as e:
            print(f"Bot: Erro de conexão ou API Key ausente. Verifique as configurações. Descrição: {e}\n")
            break

if __name__ == "__main__":
    iniciar_bot()
