# 🕵️‍♂️ Detecção de Fraudes em Cartão de Crédito (Credit Card Fraud Detection)
Este repositório contém um projeto completo de Machine Learning focado em resolver um problema clássico e desafiador do setor financeiro: a detecção de fraudes em transações de cartão de crédito. O projeto foi desenvolvido com base em um estudo prático de modelagem preditiva para dados altamente desbalanceados.

# 📋 1. O Problema e o Desafio do Desbalanceamento
O dataset utilizado contém transações realizadas por cartões de crédito em setembro de 2013 por titulares europeus.

- **O Desbalanceamento:** Apenas **0,17%** das transações na base representam fraudes (classe 1), enquanto **99,83%** são transações legítimas (classe 0).

- **Por que a acurácia engana:** Em um cenário com 99,8% de classes majoritárias, um modelo "ingênuo" que preveja sempre que uma transação é legítima atinge 99,8% de acurácia, mas falha em detectar **absolutamente nenhuma fraude**. Por isso, o foco deste projeto foi totalmente direcionado para métricas focadas na classe minoritária: **Recall**, **Precisão** e **F1-Score**.

# 🛠️ 2. Preparação e Engenharia de Dados (`Feature Engineering`)
Na etapa de preparação dos dados, as seguintes ações foram tomadas:

- **Comportamento Temporal:** Como transações fraudulentas costumam seguir padrões temporais específicos, ordenei a base por tempo (`Time`) e criei duas novas features:

- **Time_Diff:** Diferença de tempo entre a transação atual e a imediatamente anterior.

- **Amount_Rolling_Mean:** Média móvel do valor (`Amount`) em uma janela recente de transações.

- **Transformação Logarítmica:** Apliquei `np.log1p()` na coluna `Amount` para suavizar a assimetria severa dos valores monetários.

- **Padronização:** Utilizei o `StandardScaler` em todas as variáveis numéricas.

- **Divisão Estratificada:** O conjunto de dados foi dividido em treino e teste utilizando `stratify=y`, garantindo que a proporção de fraudes fosse idêntica em ambos os subsets.

# 🤖 3. Modelagem e Comparação
Trabalhamos com múltiplos algoritmos comparados pelo **Recall da fraude**:

- **Baseline:** Regressão Logística padrão.

- **Random Forest (Balanceado):** Utilizando `class_weight='balanced'`.

- **XGBoost Classifier:** Configurado com o parâmetro `scale_pos_weight` para penalizar erros na classe minoritária.

- **Otimização de Hiperparâmetros:** Uso do `GridSearchCV` no Random Forest focando em maximizar o `recall`.

# 📊 4. Ajuste de Limiar (Threshold Tuning) e Resultados
Como os modelos retornam probabilidades contínuas entre 0 e 1, o limiar padrão de 0.5 raramente é o ideal para dados desbalanceados.

- Implementamos uma função baseada na **Curva Precision-Recall** que varre os limiares de decisão para encontrar automaticamente o ponto de corte ótimo que maximiza o F1-Score (e por consequência, o Recall de fraudes).

- O relatório de classificação e as curvas ROC-AUC evidenciaram que os modelos baseados em árvores (Random Forest e XGBoost) superaram significativamente o baseline linear.

# 💡 5. Explicabilidade com SHAP
Para abrir a "caixa-preta" dos modelos, utilizamos a biblioteca **SHAP (SHapley Additive exPlanations)**.

- O gráfico de resumo do SHAP (`summary_plot`) gerado a partir de uma amostra de teste revelou quais variáveis tiveram maior peso global e individual na predição de fraudes, destacando a relevância tanto das componentes principais transformadas por PCA quanto das variáveis temporais criadas no pipeline.

# 🚀 6. O que foi evoluído em relação à proposta original
Neste projeto, além de seguir a espinha dorsal de um pipeline de Machine Learning robusto, implementamos melhorias avançadas:

- Criação de **variáveis de comportamento temporal em sequência** (`Time_Diff` e `Amount_Rolling_Mean`).

- Uso de **XGBoost** com ajuste dinâmico de `scale_pos_weight`.

- **Busca de hiperparâmetros** via `GridSearchCV` orientada a Recall.

- **Ajuste fino de limiar de decisão** (Threshold Tuning) via Curva Precision-Recall.

- Interpretabilidade avançada com **SHAP Values**.

# 💻 Como Executar o Projeto

1. Clone o repositório:

`git clone https://github.com/paulo-magls/fraud-detection-notebook.git`

2. Instale as dependências necessárias:

`pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib`

3. Abra o Jupyter Notebook e execute as células sequencialmente (o dataset será baixado automaticamente direto do link oficial do TensorFlow).