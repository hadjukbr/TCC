# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:49:54 2024

@author: alexandre gaia
"""
#%% Arvore de decisão com ajuste automatico de hiperparametros 

# desenvolvido na IDLE python Spyder 5.0

pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install imblearn
pip install time

#%% Importando as bibliotecas 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

#%% Carregando o banco de dados
data = pd.read_excel('lista_saf.xlsx')

#%% Coleta de amostra aleatória

df_sample = data.sample(frac=0.1, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero da falha
df_sample = data.tail(1000)
#%% Feature Engineering - Criando novas features baseadas em combinações de 'CCO' e 'LINHA'
df_sample['CCO_LINHA'] = df_sample['CCO'].astype(str) + "_" + df_sample['LINHA'].astype(str)

#%% Separando as variáveis independentes e a variável dependente
X = df_sample[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA','CCO_LINHA']]
y = df_sample['CANCELAMENTO_FALHA']

#%% Convertendo variáveis categóricas com LabelEncoder
le = LabelEncoder()
X = X.apply(le.fit_transform)

#%% Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#%% Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Definindo o classificador e os hiperparâmetros a serem ajustados no GridSearchCV
clf = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [5, 10, 15, 20], #profundidade máxima da árvore
    'min_samples_split': [2, 10, 20], # mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 5, 10], # mínimo de amostras presentes em nó-folha
    'criterion': ['gini', 'entropy'] # Função utilizada para medir a qualidade das divisões
}

#%% Medindo o tempo de execução e encontrando os melhores hiperparametros no GridSearchCV
start_time = time.time()

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

end_time = time.time()
execution_time = end_time - start_time

#%% Obtendo os melhores parâmetros
best_params = grid_search.best_params_
print(f"Melhores Parâmetros: {best_params}")
print(f"Tempo de Execução do GridSearchCV: {execution_time:.2f} segundos")

#%% Avaliando o modelo na base de treino com validação cruzada
best_clf = grid_search.best_estimator_

# Validação cruzada na base de treino
cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f'Acurácia média com Validação Cruzada na Base de Treino: {cv_scores.mean():.6f}')

#%% Avaliando o desempenho na base de treino
y_train_pred = best_clf.predict(X_train)
y_train_prob = best_clf.predict_proba(X_train)[:, 1]

accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

print(f"\nMétricas para a Base de Treino:")
print(f"Acurácia (Treino): {accuracy_train:.6f}")
print(f"Precisão (Treino): {precision_train:.6f}")
print(f"Recall (Treino): {recall_train:.6f}")
print(f"F1-Score (Treino): {f1_train:.6f}")

#%% Avaliando o desempenho na base de teste
y_test_pred = best_clf.predict(X_test)
y_test_prob = best_clf.predict_proba(X_test)[:, 1]

accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print(f"\nMétricas para a Base de Teste:")
print(f"Acurácia (Teste): {accuracy_test:.6f}")
print(f"Precisão (Teste): {precision_test:.6f}")
print(f"Recall (Teste): {recall_test:.6f}")
print(f"F1-Score (Teste): {f1_test:.6f}")

#%% Matriz de Confusão para a Base de Teste
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nMatriz de Confusão (Teste):\n", conf_matrix)

# Plotando a Matriz de Confusão
plt.figure(figsize=(6, 4))
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Cancelado', 'Cancelado'], yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão - Base de Teste')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()


#%% Curva ROC para a Base de Teste
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plotando a Curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.6f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Teste')
plt.legend(loc="lower right")
plt.show()

    #%% Desenho da árvore de decisão
plt.figure(figsize=(200,100))  # Ajustar o tamanho da figura conforme necessário
plot_tree(best_clf, feature_names=X.columns, class_names=['Não Cancelado', 'Cancelado'], filled=True, rounded=True, fontsize=25)
plt.title("Árvore de Decisão")
plt.show()


