# ClinicaIA — Sistema de Predição de Risco Clínico

Este projeto é uma aplicação completa e funcional de Machine Learning aplicada à área da saúde. Ele integra um banco de dados relacional (SQLite), um modelo de Inteligência Artificial treinado (Random Forest) e uma interface interativa (Streamlit) para auxiliar médicos na triagem de pacientes.

O sistema atende a todos os requisitos propostos na "Fase 3: Integração e Interface", unindo os scripts de laboratório em um produto robusto e didático.

## 🚀 Funcionalidades Principais

- **Autenticação Segura:** Tela de login validada diretamente no banco de dados SQLite (com hash de senhas).
- **Dashboard Analítico:** Visão geral do sistema, total de pacientes, exames realizados e distribuição de riscos.
- **Predição por IA em Tempo Real:** O médico insere os dados do paciente (glicose, pressão, IMC e colesterol) e o modelo Random Forest retorna imediatamente o grau de risco (Normal, Alerta ou Alto Risco) com a porcentagem de confiança.
- **Gestão de Exames:** Permite cadastrar e configurar novos tipos de exames, definindo os limiares para "Alerta" e "Alto Risco".
- **Cadastro e Histórico de Pacientes:** Registro de novos pacientes e visualização completa do histórico de exames de cada um.
- **Feedback Visual:** Alertas coloridos e claros indicando se cada parâmetro está dentro da normalidade ou se exige atenção médica imediata.

## 🛠️ Stack Tecnológica

- **Linguagem:** Python 3.10+
- **Front-end:** Streamlit
- **Banco de Dados:** SQLite3
- **Machine Learning:** Scikit-Learn (RandomForestClassifier)
- **Processamento de Dados:** Pandas, NumPy
- **Serialização do Modelo:** Joblib

## 📂 Estrutura do Projeto

O arquivo `.zip` contém os seguintes componentes:

```text
clinica_risco/
├── app.py                   # Interface principal em Streamlit (Front-end)
├── setup_db.py              # Script para criação do banco SQLite e importação do CSV
├── treinar_modelo.py        # Script para treinamento e avaliação do modelo de ML
├── testar_sistema.py        # Script de testes automatizados de todos os componentes
├── pacientes.csv            # Base de dados simulada (2.000 registros)
├── clinica.db               # Banco de dados relacional pronto para uso
├── modelo_risco.pkl         # Modelo de IA treinado (Acurácia de 93,5%)
└── graficos/                # Gráficos gerados durante a análise e avaliação do modelo
```

## ⚙️ Como Executar Localmente

### 1. Instalar as Dependências

Abra o terminal na pasta do projeto e instale as bibliotecas necessárias:

```bash
pip install pandas numpy scikit-learn joblib streamlit seaborn matplotlib
```

### 2. Configurar o Banco de Dados e Treinar o Modelo (Opcional)

O projeto já vai com o banco `clinica.db` populado e o `modelo_risco.pkl` treinado. No entanto, se você quiser recriar tudo do zero, execute:

```bash
# Cria o banco e importa os dados do CSV
python setup_db.py

# Treina o modelo de IA e gera os gráficos de avaliação
python treinar_modelo.py
```

### 3. Iniciar a Aplicação

Para abrir a interface do sistema, execute o comando abaixo:

```bash
streamlit run app.py
```

O sistema abrirá automaticamente no seu navegador padrão (geralmente em `http://localhost:8501`).

### 4. Credenciais de Acesso

Para entrar no sistema, utilize um dos usuários padrão:

- **Usuário:** `admin` | **Senha:** `admin123`
- **Usuário:** `medico` | **Senha:** `med2024`

## 📊 Desempenho do Modelo de IA

O modelo foi treinado utilizando o algoritmo **Random Forest** com validação cruzada estratificada (5-fold). Os resultados no conjunto de teste foram excelentes:

- **Acurácia Global:** 93,5%
- **Precisão para Alto Risco:** 97%
- **Recall para Alto Risco:** 93%

Os gráficos de matriz de confusão e importância das variáveis estão disponíveis na pasta `graficos/`.

---
*Projeto desenvolvido como demonstração acadêmica de integração entre Ciência de Dados, Engenharia de Software e Saúde Digital Health.*
