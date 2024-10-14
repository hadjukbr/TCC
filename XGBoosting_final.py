# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:52:21 2024

@author: alexandre gaia
"""
#%% Modelo de Extreme Gradient Boosting 
# Instalando os pacotes necessários 
!pip install xgboost
!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install statsmodels
!pip install scikit-learn
!pip install imblearn
!pip install time

#%% Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, roc_auc_score, confusion_matrix, 
                             classification_report, log_loss, cohen_kappa_score)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import time

#%% Carregando os dados
data = pd.read_excel('lista_saf.xlsx')

#%% Amostragem de x% dos dados 
data_amostra = data.sample(frac=0.1, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero da falha
data_amostra = data.tail(1000)
#%% Separando as variáveis independentes e a variável alvo
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#%% Aplicando Label Encoding para variáveis categóricas
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
    

#%% Aplicando Random Over-Sampling para lidar com o desbalanceamento
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

#%% Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Definindo o modelo base e o GridSearchCV para buscar os melhores hiperparâmetros
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

# Definindo os parâmetros a serem otimizados
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 500], # qtde. de árvores do modelo 
    'max_depth': [4, 6, 8], # profundidade máxima da árvore
    'learning_rate': [0.01, 0.1, 0.2], # taxa de aprendizado
    'subsample': [0.8, 1.0], # proporção de amostras utilizadas 
    'colsample_bytree': [0.8, 1.0] # proporção de variáveis escolhidas aleatoriamente 
}

# Configurando o GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

# Treinando o modelo com a busca pelos melhores hiperparâmetros
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()
execution_time = end_time - start_time

# Mostrar o tempo de execução
print(f"Tempo de execução: {execution_time:.2f} segundos") #Tempo calculo para 
#comparar o poder computacional de cada modelo 

# Exibindo os melhores hiperparâmetros
print("Melhores hiperparâmetros encontrados:", grid_search.best_params_)

# Melhor modelo encontrado pelo GridSearchCV
best_model = grid_search.best_estimator_

#%% Fazendo previsões
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
y_train_prob = best_model.predict_proba(X_train)[:, 1]
y_test_prob = best_model.predict_proba(X_test)[:, 1]

#%% Avaliação do Modelo - Base de Treino
print("\n--- Avaliação na Base de Treino ---")
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
log_loss_train = log_loss(y_train, y_train_prob)
kappa_train = cohen_kappa_score(y_train, y_train_pred)

print(f'Acurácia: {accuracy_train:.4f}')
print(f'Precisão: {precision_train:.4f}')
print(f'Recall: {recall_train:.4f}')
print(f'F1-Score: {f1_train:.4f}')
print(f'Log Loss: {log_loss_train:.4f}')
print(f'Cohen\'s Kappa: {kappa_train:.4f}')

# Relatório de Classificação - Treino
print("\nRelatório de Classificação (Treino):\n", classification_report(y_train, y_train_pred))

#%% Avaliação do Modelo - Base de Teste
print("\n--- Avaliação na Base de Teste ---")
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
log_loss_test = log_loss(y_test, y_test_prob)
kappa_test = cohen_kappa_score(y_test, y_test_pred)

print(f'Acurácia: {accuracy_test:.4f}')
print(f'Precisão: {precision_test:.4f}')
print(f'Recall: {recall_test:.4f}')
print(f'F1-Score: {f1_test:.4f}')
print(f'Log Loss: {log_loss_test:.4f}')
print(f'Cohen\'s Kappa: {kappa_test:.4f}')

# Relatório de Classificação - Teste
print("\nRelatório de Classificação (Teste):\n", classification_report(y_test, y_test_pred))

#%% Matriz de Confusão - Teste
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Cancelado', 'Cancelado'], yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão - Teste')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()

    #%% Curva ROC - Teste
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (área = %0.6f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Teste')
plt.legend(loc="lower right")
plt.show()

#%% Verificando Overfitting
print("\n--- Verificação de Overfitting ---")
print(f"Acurácia no treino: {accuracy_train:.4f}")
print(f"Acurácia no teste: {accuracy_test:.4f}")

if accuracy_train > accuracy_test + 0.05:
    print("Possível overfitting detectado! A acurácia de treino é significativamente maior que a de teste.")
else:
    print("Nenhum sinal de overfitting evidente.")
#%% Plotagem das duas curvas ROC
#%% Curva ROC para o conjunto de treino
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
roc_auc_train = roc_auc_score(y_train, y_train_prob)

# Curva ROC para o conjunto de teste
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = roc_auc_score(y_test, y_test_prob)

#%% Plotando as curvas ROC para treino e teste
plt.figure()

# Curva ROC para o conjunto de treino
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Treino (area = {roc_auc_train:.6f})')

# Curva ROC para o conjunto de teste
plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Teste (area = {roc_auc_test:.6f})')

# Linha base
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Ajustes no gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Treino vs Teste')
plt.legend(loc="lower right")

# Exibindo o gráfico
plt.show()