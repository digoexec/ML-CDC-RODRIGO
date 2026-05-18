# 🤖 Exercícios de Chatbots e LLMs — Guia de Estudo

## Pré-requisitos

```bash
pip install pandas scikit-learn anthropic
```

Coloque o arquivo `logs_ecommerce.csv` na mesma pasta dos scripts.

---

## Mapa dos Exercícios

| # | Arquivo | Conceito Principal | Tecnologia |
|---|---------|-------------------|------------|
| 1 | `ex01_classificador_tfidf.py` | Classificação de Intenções | TF-IDF + Naive Bayes |
| 2 | `ex02_memoria_contexto.py` | Session State / Memória | Python OOP |
| 3 | `ex03_regex_entidades.py` | Extração de Entidades | Regex (`re`) |
| 4 | `ex04_sentimento_transbordo.py` | Análise de Sentimento + Escalation | Léxico + Regras |
| 5 | `ex05_fsm_estados.py` | Máquina de Estados Finita | FSM Pattern |
| 6 | `ex06_faq_cosseno.py` | Busca Semântica | Cosseno + CountVectorizer |
| 7 | `ex07_rag_csv.py` | RAG Primitivo | Pandas + Templates |
| 8 | `ex08_orquestrador_tools.py` | Tool Use / Roteamento | Router Pattern |
| 9 | `ex09_api_llm_persona.py` | API LLM + System Prompt | Claude API |

---

## Como executar cada um

```bash
python ex01_classificador_tfidf.py
python ex02_memoria_contexto.py
# ... e assim por diante
```

---

## Conceitos-chave por exercício

### Ex1 — TF-IDF + Naive Bayes
- **TF-IDF**: transforma texto em vetores numéricos ponderados por relevância
- **Naive Bayes**: aprende probabilidades P(intenção | palavras)
- Substitui if/else por um modelo que generaliza para frases nunca vistas

### Ex2 — Memória de Curto Prazo
- `self.contexto` é o Session State que persiste entre turnos
- Sem isso, cada resposta seria independente (amnésia)
- Em produção: Redis, banco de dados, ou variáveis de sessão web

### Ex3 — Regex / NER
- `r'#\d{4}'` = padrão que casa com #1234, #5678...
- `re.findall()` retorna todas as ocorrências no texto
- Base de qualquer sistema de triagem automatizada

### Ex4 — Sentimento + Transbordo
- Léxico com pesos por palavra → score de irritação
- Threshold ≥ 2 → transbordo humano (escalation)
- Human-in-the-Loop é obrigatório em sistemas de produção

### Ex5 — FSM
- Estados definem o que o usuário PODE fazer em cada momento
- Transições são disparadas por inputs válidos
- Previne conversas incoerentes e inputs inesperados

### Ex6 — Similaridade de Cosseno
- Vetores de palavras → ângulo entre eles = similaridade
- Funciona mesmo com palavras diferentes (sinônimos parciais)
- Base dos sistemas de busca semântica modernos

### Ex7 — RAG
- Retrieve: busca dados no CSV por ID
- Augment: injeta esses dados no template de resposta
- Base de todos os assistentes empresariais modernos

### Ex8 — Tool Use / Router
- Bot detecta intenção → seleciona a função correta → executa
- Exatamente como GPT-4/Claude fazem Function Calling
- Padrão Orquestrador: o LLM coordena, as ferramentas executam

### Ex9 — API LLM com Persona
- System Prompt define quem o bot "é" e como se comporta
- Histórico multi-turn: toda conversa é enviada a cada nova requisição
- É assim que ChatGPT/Claude.ai funcionam por baixo dos panos
