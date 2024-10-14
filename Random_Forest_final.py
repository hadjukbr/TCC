# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:48:38 2024

@author: alexandre gaia
"""
#%% Modelo algortimo de Random Forest (florestas aleatorias)
# Modelo pode ser utilizado para medir a importância de cada variavel utilizada 
# Instalando as bibliotecas utilizadas - algortimo desenvolvido no Spyder 

pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install imblearn
pip install time

#%% Importando as bibliotecas necessárias

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time

#%% Carregando o banco de dados
data = pd.read_excel('lista_saf.xlsx')

#%% Coleta de amostra aleatória
data_Amostra = data.sample(frac=0.1, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero da falha
data_amostra = data.tail(1000)

#%% Separar as features e o target
X = data_Amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'ID_GRSIST_FALHA', 'LOCALIDADE_FALHA']]
y = data_Amostra['CANCELAMENTO_FALHA']

#%% Label Encoding para as variáveis categóricas
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col])

#%% Aplicar o SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#%% Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Definindo os hiperparâmetros para a busca
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 500], # número de árvores da floresta
    'max_depth': [5, 10, 20, 30, None], # Profundidade máxima da floresta
    'min_samples_split': [2, 5, 10], # mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4, 5] # mínimo de amostras presentes em nó-folha
}

#%% Iniciar o GridSearchCV com RandomForest
rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV, usando 3-fold cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

# Medir o tempo de execução
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()
execution_time = end_time - start_time

# Mostrar o tempo de execução
print(f"Tempo de execução: {execution_time:.2f} segundos")

#%% Melhor modelo e hiperparâmetros
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Melhores hiperparâmetros: {best_params}")

#%% Avaliação na base de treino
y_train_pred = best_model.predict(X_train)  # Previsões na base de treino
y_train_prob = best_model.predict_proba(X_train)[:, 1]

# Avaliar métricas na base de treino
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print("\nResultados da base de TREINO:")
print(f'Acurácia: {train_accuracy:.6f}')
print(f'Precisão: {train_precision:.6f}')
print(f'Revocação: {train_recall:.6f}')
print(f'F1-Score: {train_f1:.6f}')
print("\nRelatório de Classificação - TREINO:")
print(classification_report(y_train, y_train_pred))

#%% Avaliação na base de teste
y_test_pred = best_model.predict(X_test)  # Previsões na base de teste
y_test_prob = best_model.predict_proba(X_test)[:, 1]

# Avaliar métricas na base de teste
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\nResultados da base de TESTE:")
print(f'Acurácia: {test_accuracy:.6f}')
print(f'Precisão: {test_precision:.6f}')
print(f'Revocação: {test_recall:.6f}')
print(f'F1-Score: {test_f1:.6f}')
print("\nRelatório de Classificação - TESTE:")
print(classification_report(y_test, y_test_pred))

#%% Matriz de Confusão na base de teste
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - TESTE')
plt.show()

    #%% Curva ROC na base de teste
        
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (área = %0.6f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - TESTE')
plt.legend(loc="lower right")
plt.show()

#%% Importância das Features
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

print("Importância das Features:")
for idx in indices:
    print(f"{features[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.title("Importância das Features")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

#%% Verificar Overfitting
print("\n--- Verificação de Overfitting ---")
if train_accuracy > test_accuracy + 0.05:
    print("Possível overfitting detectado! A acurácia de treino é significativamente maior que a de teste.")
else:
    print("Nenhum sinal de overfitting evidente.")

# Comparar as métricas de treino e teste
print(f"\nAcurácia no treino: {train_accuracy:.6f}")
print(f"Acurácia no teste: {test_accuracy:.6f}")
