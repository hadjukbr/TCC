# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:44:57 2024

@author: alexandre gaia
"""

#%% Instalando o TPOT 
pip install tpot
pip install sklearn
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install imblearn
pip install time

https://epistasislab.github.io/tpot/installing/ Passo a passo para instalação
#Possivelmente haverá conflitos entre as versões dos pacotes utilizados (SMOTE,TPOT e sklearn)
#Mantenha os pacotes atualizados 
# Caso não tenha os pacotes Pytorch, Cuml e outros que utilizem GPU, não serão considerados na pesquisa
# Utilize equipamento com alto grau de processamento para diminuir o tempo de execução
# Pode interromper a execução, porém não terá total otimização

#O TPOT é uma ferramenta de AutoML (Automated Machine Learning) que automatiza
# o processo de construção de modelos preditivos. Ele usa um algoritmo genético
# para testar diferentes combinações de pipelines de machine learning, ajustando 
#hiperparâmetros e escolhendo a melhor solução com base no desempenho. O TPOT é
# uma boa escolha se você deseja economizar tempo na seleção e ajuste de modelos.

#%%# Importar as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

#%% Carregar o dataset
data = pd.read_excel('lista_saf.xlsx')

# Coleta de amostra aleatória
data_amostra = data.sample(frac=0.1, random_state=42)

#%%Amostra das ultimas observações do data frame(registros mais recentes)

data = data.sort_values(by = ['ANO','NRFALHA'], ascending=True ) #realizando o ordenamento da SAF pelo numero da falha
data_amostra = data.tail(1000)
#%% Separar as features e o target
X = data_amostra[['CCO', 'NIVEL', 'LINHA', 'TIPO_GRSIST_FALHA', 'TX_TIPO_LOCALIDADE', 'LOCALIDADE_FALHA', 'ID_GRSIST_FALHA']]
y = data_amostra['CANCELAMENTO_FALHA']

#%% Converter as variáveis categóricas em numéricas (Label Encoding)
label_encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':  # Se for uma variável categórica
        X[col] = label_encoder.fit_transform(X[col])

#%% Balancear as classes usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#%% Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#%% Inicializar o TPOT com limite de tempo para otimização

tpot = TPOTClassifier(verbosity=2, generations=5, population_size=50, random_state=42, scoring='accuracy')

#%% Treinar o modelo com TPOT

start_time = time.time()  # Medir tempo de execução
tpot.fit(X_train, y_train)
#Exibir tempo de execução do algoritmo
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução do algoritmo: {execution_time:.2f} segundos")

#%% Avaliar o desempenho no conjunto de teste
y_pred = tpot.predict(X_test)
y_prob = tpot.predict_proba(X_test)[:, 1]  # Probabilidades para a Curva ROC

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.6f}")
print(f"Precisão: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-Score: {f1:.6f}")

#%% Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

#%% Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

#%% Curva ROC e AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.6f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Modelo Otimizado TPOT')
plt.legend(loc="lower right")
plt.show()


#%% Exportar o pipeline final escolhido pelo TPOT
tpot.export('best_model_pipeline1k.py')
