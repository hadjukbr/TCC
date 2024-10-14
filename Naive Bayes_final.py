# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 01:50:42 2024

@author: alexandre gaia
"""
# Modelo de Naive Bayes com variações e ajustes de hiperparametros 
# Modelo utilizado para testar com amostras inferiores a 10%, porém sendo possivel
# utilizar frações maiores  

#Modelo descartado após análise final dos resultados 
#Utilize as mesmas bibliotecas dos outros modelos preditivos 
#%% Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import (GaussianNB, CategoricalNB, ComplementNB, 
                                 BernoulliNB, MultinomialNB)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, ConfusionMatrixDisplay, matthews_corrcoef, 
                             log_loss, brier_score_loss, cohen_kappa_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time

#%% Carregando a base de dados
data = pd.read_excel('lista_saf.xlsx')

# Coletando uma amostra aleatória de x% dos dados
data_amostra = data.sample(frac=0.025, random_state=42)

#%% Separar as features e o target
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'ID_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#%% Aplicar Label Encoding para variáveis categóricas
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
    
#%% Aplicando o SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#%% Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Após a divisão dos dados, podemos testar modelos diferentes de Naive Bayes 
# Cada um com parametros diferentes e objetivos diferentes 
# Para efeito de estudos, iremos testar todas a variações com até 10% do Data Frame 
#%% Definindo o modelo Naive Bayes Gaussiano e os hiperparâmetros para o GridSearch
nb = GaussianNB()
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100)  # Hiperparâmetro de suavização
}

#%% Definindo o modelo Bernoulli Naive Bayes  e os hiperparâmetros para o GridSearch
nb = BernoulliNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0],      # Testar diferentes valores de suavização
    'binarize': [0.0, 0.5, 1.0],   # Testar diferentes valores para binarizar as características
    'fit_prior': [True, False]     # Testar se ajusta as probabilidades a priori ou não
}

#%% Definindo o modelo Naive Bayes Multinomial e os hiperparâmetros para o GridSearch

nb = MultinomialNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Testar diferentes valores de suavização
    'fit_prior': [True, False]       # Testar se ajusta as probabilidades a priori ou não
}

#%% Definindo o modelo Naive Bayes Categórico e os hiperparâmetros para o GridSearch
nb = CategoricalNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Testar diferentes valores de suavização
    'fit_prior': [True, False]       # Testar se ajusta as probabilidades a priori ou não
}

#%%#%% Definindo o modelo Naive Bayes Complementar e os hiperparâmetros para o GridSearch

nb = ComplementNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Testar diferentes valores de suavização
    'norm': [True, False],          # Testar se deve normalizar os dados ou não
    'fit_prior': [True, False]      # Testar se ajusta as probabilidades a priori ou não
}

#%% Realizando a busca pelos melhores hiperparâmetros com o GridSearchCV
grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1, scoring= 'accuracy')

#%% Medindo o tempo de execução
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()
execution_time = end_time - start_time

# Exibindo o tempo de execução
print(f"Tempo de execução: {execution_time:.2f} segundos")

# Melhor modelo e parâmetros
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros: {best_params}")

#%% Avaliação na base de treino
y_train_pred = best_model.predict(X_train)
y_train_prob = best_model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_report = classification_report(y_train, y_train_pred)

print("\nResultados da base de TREINO:")
print(f'Acurácia: {train_accuracy:.6f}')
print(train_report)

# Calculando métricas adicionais na base de treino
train_mcc = matthews_corrcoef(y_train, y_train_pred)
train_log_loss = log_loss(y_train, y_train_prob)
train_brier = brier_score_loss(y_train, y_train_prob)
train_kappa = cohen_kappa_score(y_train, y_train_pred)

print(f"MCC (Treino): {train_mcc:.6f}")
print(f"Log Loss (Treino): {train_log_loss:.6f}")
print(f"Brier Score (Treino): {train_brier:.6f}")
print(f"Cohen's Kappa (Treino): {train_kappa:.6f}")

#%% Avaliação na base de teste
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)

print("\nResultados da base de TESTE:")
print(f'Acurácia: {test_accuracy:.6f}')
print(test_report)

# Calculando métricas adicionais na base de teste
test_mcc = matthews_corrcoef(y_test, y_test_pred)
test_log_loss = log_loss(y_test, y_test_prob)
test_brier = brier_score_loss(y_test, y_test_prob)
test_kappa = cohen_kappa_score(y_test, y_test_pred)

print(f"MCC (Teste): {test_mcc:.6f}")
print(f"Log Loss (Teste): {test_log_loss:.6f}")
print(f"Brier Score (Teste): {test_brier:.6f}")
print(f"Cohen's Kappa (Teste): {test_kappa:.6f}")

#%% Verificando Overfitting
print("\n--- Verificação de Overfitting ---")
if train_accuracy > test_accuracy + 0.05:
    print("Possível overfitting detectado! A acurácia de treino é significativamente maior que a de teste.")
else:
    print("Nenhum sinal de overfitting evidente.")

# Comparar as métricas de treino e teste
print(f"Acurácia no treino: {train_accuracy:.6f}")
print(f"Acurácia no teste: {test_accuracy:.6f}")

#%% Matriz de Confusão - Base de Treino
cm_train = confusion_matrix(y_train, y_train_pred)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
disp_train.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - TREINO')
plt.show()

#%% Matriz de Confusão - Base de Teste
cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp_test.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - TESTE')
plt.show()
#%% Curva ROC e AUC - Base de Treino
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Curva ROC - Treino (área = %0.6f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - TREINO')
plt.legend(loc="lower right")
plt.show()

#%% Curva ROC e AUC - Base de Teste
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='Curva ROC - Teste (área = %0.6f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - TESTE')
plt.legend(loc="lower right")
plt.show()
