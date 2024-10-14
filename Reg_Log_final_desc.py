# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:47:32 2024

@author: alexandre gaia
"""
#%% Modelo de Regressão linear com calculo de logito 
# Desenvolvido para testes para verificar o comportamento dos dados 
# Descartado do resultado final devido ao pessimo desempenho visualizado 

#Desenvolvido na IDLE Spyder 5.0

#Instalando as bibliotecas necessarias caso não tenha feito anteriormente 
pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
pip install imblearn
pip install time


#%% Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import time  # Biblioteca para medir o tempo de execução
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

#%% Carregar o banco de dados
data = pd.read_excel('lista_saf.xlsx')

# Coleta de amostra aleatória
data_amostra = data.sample(frac=0.01, random_state=42)

#%% Separar as features e o target
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#%% Aplicar Label Encoding para variáveis categóricas
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
    
#%% Aplicar o SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#%% Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Ajuste fino dos hiperparâmetros com GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 3}, {0: 1, 1: 5}]
}

log_reg = LogisticRegression(random_state=42)

# Iniciar a medição de tempo
start_time = time.time()

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Calcular o tempo de execução
end_time = time.time()
execution_time = end_time - start_time

# Exibir o tempo de execução
print(f"Tempo de execução do GridSearchCV: {execution_time:.2f} segundos")

#%% Avaliar os melhores hiperparâmetros
best_params = grid_search.best_params_
print(f"Melhores Hiperparâmetros: {best_params}")

#%% Treinar o modelo com os melhores hiperparâmetros
model_logit = LogisticRegression(**best_params, random_state=42)
model_logit.fit(X_train, y_train)

#%% Fazer previsões nos dois conjuntos (treino e teste)
y_train_pred = model_logit.predict(X_train)
y_test_pred = model_logit.predict(X_test)

#%% Probabilidades
y_train_prob = model_logit.predict_proba(X_train)[:, 1]
y_test_prob = model_logit.predict_proba(X_test)[:, 1]

#%% Avaliação dos Resultados - Conjunto de Treino
print("Métricas no Conjunto de Treino:")
print(f"Acurácia: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precisão: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"F1-Score: {f1_score(y_train, y_train_pred):.4f}")

# Matriz de Confusão - Treino
cm_train = confusion_matrix(y_train, y_train_pred)
print(f"\nMatriz de Confusão - Treino:\n{cm_train}")

# Curva ROC - Treino
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Treino ROC curve (area = {roc_auc_train:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Conjunto de Treino')
plt.legend(loc="lower right")
plt.show()

#%% Avaliação dos Resultados - Conjunto de Teste
print("Métricas no Conjunto de Teste:")
print(f"Acurácia: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")

# Matriz de Confusão - Teste
cm_test = confusion_matrix(y_test, y_test_pred)
print(f"\nMatriz de Confusão - Teste:\n{cm_test}")

# Curva ROC - Teste
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Teste ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Conjunto de Teste')
plt.legend(loc="lower right")
plt.show()

#%% Validação Cruzada no Conjunto de Treino
cv_scores = cross_val_score(model_logit, X_train, y_train, cv=5, scoring='accuracy')
print(f"Acurácia Média com Cross-Validation no Treino: {np.mean(cv_scores):.4f}")

#%% Calcular o logito
def calcular_logito(observations, model):
    logit_value = np.dot(observations, model.coef_[0]) + model.intercept_[0]
    return logit_value

#%% Função para testar com dados inseridos pelo usuário
def test_observations():
    print("Digite os valores para as variáveis preditoras:")

    # Coleta de dados do usuário
    cco = input("CCO: ")
    nivel = input("NIVEL: ")
    linha = input("LINHA: ")
    tipo_grsist_falha = input("TIPO_GRSIST_FALHA: ")
    id_grsist_falha = input("ID_GRSIST_FALHA: ")
    tx_tipo_localidade = input("TX_TIPO_LOCALIDADE: ")
    localidade_falha = input("LOCALIDADE_FALHA: ")

    # Criar um DataFrame com a observação do usuário
    new_obs = pd.DataFrame([[cco, nivel, linha, tipo_grsist_falha, id_grsist_falha, tx_tipo_localidade, localidade_falha]],
                           columns=['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA'])

    # Transformar as variáveis categóricas com as mesmas categorias do treinamento
    for column in new_obs.columns:
        new_obs[column] = label_encoders[column].transform(new_obs[column].astype(str))

    # Normalizar a nova observação
    new_obs_scaled = scaler.transform(new_obs)

    # Calcular o logito
    logit_value = calcular_logito(new_obs_scaled, model_logit)
    print(f"Logito calculado: {logit_value[0]}")

    # Calcular a probabilidade
    prob = model_logit.predict_proba(new_obs_scaled)[:, 1]
    print(f"Probabilidade de CANCELAMENTO_FALHA = 1: {prob[0]}")

    # Prever a classe
    predicted_class = model_logit.predict(new_obs_scaled)
    print(f"Classe predita: {predicted_class[0]}")

#%% Testar com dados inseridos pelo usuário
test_observations()
