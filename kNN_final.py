# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:40:37 2024

@author: alexandre gaia
"""

#%% Modelo de K-Nearest Neighbors - K-vizinhos mais próximos 
# Instalando os pacotes necessários 
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
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

#%% Carregar os dados
data = pd.read_excel('lista_saf.xlsx')

#%% Coleta de amostra aleatória
data_amostra = data.sample(frac=0.05, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero 
data_amostra = data.tail(5000)
#%% Separar as features e o target
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#Removendo a variavel  ID_GRSIST_FALHA  que apresentou VIF 12.192567
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
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Reduzindo a dimensionalidade dos dados 
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Mantém 95% da variação dos dados
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#Verificar os resultados após a execuaçao do PCA 
#Resultado apresentado no TCC não inclui PCA, pois reduziu a acuracia na base de treino 
# com forte tendencia pra overfitting 
#%% Pipeline para tentar corrigir erro no KNN 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

#%% GridSearchCV para encontrar os melhores hiperparâmetros (n_neighbors e distância)
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11], # nº de vizinhos a serem testados
    'knn__weights': ['uniform', 'distance'], #ponderação em relação ao peso de cada vizinho
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev','cosine','minkowski'] #metricas de distância
}

#knn = KNeighborsClassifier()

#%% CV Estratificada : garantindo que cada fold tenha uma distribuição 
#semelhante das classes

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
#%% Cálculo do tempo de execução do algoritmo
start_time = time.time()

# Usar GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', error_score='raise')
grid_search.fit(X_train, y_train)

end_time = time.time()

#%% Tempo de execução
execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time:.2f} segundos")

# Melhor combinação de hiperparâmetros encontrada
best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros: {best_params}")

# Obter os melhores parâmetros encontrados
best_params = grid_search.best_params_

# Remover o prefixo 'knn__' dos melhores parâmetros
best_params = {key.split('__')[1]: value for key, value in best_params.items()}
    #%% Treinar o modelo kNN com os melhores hiperparâmetros
model_knn = KNeighborsClassifier(**best_params)
model_knn.fit(X_train, y_train)

#%% Fazer previsões
y_train_pred = model_knn.predict(X_train)
y_test_pred = model_knn.predict(X_test)
y_test_prob = model_knn.predict_proba(X_test)[:, 1]

#%% Avaliação do Modelo (Acurácia, Precisão, Recall, F1-score)
# Base de treino
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

# Base de teste
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

# Exibindo as métricas
print(f"Treino - Acurácia: {accuracy_train:.6f}, Precisão: {precision_train:.6f}, Recall: {recall_train:.6f}, F1-score: {f1_train:.6f}")
print(f"Teste - Acurácia: {accuracy_test:.6f}, Precisão: {precision_test:.6f}, Recall: {recall_test:.6f}, F1-score: {f1_test:.6f}")

# Relatório de Classificação para base de teste
print("\nRelatório de Classificação - Base de Teste:")
print(classification_report(y_test, y_test_pred))

#%% Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Cancelado', 'Cancelado'], yticklabels=['Não Cancelado', 'Cancelado'])
plt.title('Matriz de Confusão - Base de Teste')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()

        #%% Curva ROC e AUC para base de teste
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.6f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Base de Teste')
plt.legend(loc="lower right")
plt.show()

#%% Validação Cruzada (Cross-Validation) - Base Completa (Reamostrada)
cv_scores = cross_val_score(model_knn, X_resampled, y_resampled, cv=5, scoring='accuracy')

print(f"Acurácia Média com Cross-Validation: {np.mean(cv_scores):.6f}")
print(f"Desvio Padrão da Acurácia com Cross-Validation: {np.std(cv_scores):.6f}")
    