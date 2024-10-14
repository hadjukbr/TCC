# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 04:03:50 2024

@author: alexandre gaia
"""


#%% Modelo MLP (Multilayer perceptron)

#O Multilayer Perceptron (MLP) é um tipo de rede neural artificial composto por
# várias camadas de neurônios, sendo uma das arquiteturas mais básicas e amplamente
# utilizadas em aprendizado de máquina. 

# Modelo desenvolvido na idle spyder 

pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install imblearn
pip install time
pip install -U seaborn

#%% Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
import time
#%% Carregar os dados
data = pd.read_excel('lista_saf.xlsx')

# Coleta de amostra aleatória
data_amostra = data.sample(frac=0.1, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero da falha
data_amostra = data.tail(1000)
#%% Separar as features e o target
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'ID_GRSIST_FALHA',  'LOCALIDADE_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#%% Aplicar Label Encoding para variáveis categóricas
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

#%% Padronizar (normalizar) as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Balancear as classes com SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

#%% Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#%% Ajuste de hiperparâmetros com GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 100)],# tamanho e quantidade de camadas escondidas
    'activation': ['relu', 'tanh'], # funções de ativação
    'solver': ['adam', 'sgd'],  # algoritmos de otimização
    'alpha': [0.0001, 0.001, 0.01],  # Regularização L2
    'learning_rate': ['constant', 'adaptive']  # taxa de aprendizado
#    'learning_rate_init': [0.001, 0.01, 0.1], # taxa inicial de aprendizado - não utilizado no TCC, impacta no tempo de execução 
#   'max_iter': [200, 400, 600] # número máximo de iterações - idem anterior
}

grid_search = GridSearchCV(MLPClassifier(max_iter=1500), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Medir tempo de execução
start_time = time.time()  
grid_search.fit(X_train, y_train)
# Exibir tempo de execução do algoritmo
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução do algoritmo: {execution_time:.2f} segundos")

# Melhor combinação de hiperparâmetros encontrada
best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros: {best_params}")

#%% Reajustar o modelo com os melhores hiperparâmetros
mlp_best = MLPClassifier(**best_params, max_iter=1500, random_state=42)
mlp_best.fit(X_train, y_train)

#%% Previsões na base de treino
y_train_pred = mlp_best.predict(X_train)
y_train_prob = mlp_best.predict_proba(X_train)[:, 1]

# Previsões na base de teste
y_test_pred = mlp_best.predict(X_test)
y_test_prob = mlp_best.predict_proba(X_test)[:, 1]

#%% Avaliação do Modelo na base de treino (Acurácia, Precisão, Recall, F1)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

# Avaliação do Modelo na base de teste
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

#%% Exibir as métricas da base de treino
print("\n### Base de Treino ###")
print(f"Acurácia: {accuracy_train:.6f}")
print(f"Precisão: {precision_train:.6f}")
print(f"Recall: {recall_train:.6f}")
print(f"F1-Score: {f1_train:.6f}")
print("\nRelatório de Classificação - Treino:")
print(classification_report(y_train, y_train_pred))

#%% Exibir as métricas da base de teste
print("\n### Base de Teste ###")
print(f"Acurácia: {accuracy_test:.6f}")
print(f"Precisão: {precision_test:.6f}")
print(f"Recall: {recall_test:.6f}")
print(f"F1-Score: {f1_test:.6f}")
print("\nRelatório de Classificação - Teste:")
print(classification_report(y_test, y_test_pred))

#%% Matriz de Confusão - Treino
plt.figure(figsize=(6, 4))
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Cancelado', 'Cancelado'], yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão - Base de Treino')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()

#%% Matriz de Confusão - Teste
plt.figure(figsize=(6, 4))
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Cancelado', 'Cancelado'], yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão - Base de Teste')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()

    #%% Curva ROC e AUC - Treino
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'ROC curve (área = {roc_auc_train:.6f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Base de Treino')
plt.legend(loc="lower right")
plt.show()

#%% Curva ROC e AUC - Teste
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'ROC curve (área = {roc_auc_test:.6f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Base de Teste')
plt.legend(loc="lower right")
plt.show()
