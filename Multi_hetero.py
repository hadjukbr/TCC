# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:18:08 2024

@author: alexandre gaia
"""
#%% Essa interação foi utilizada devido a incompatibilidade do pacote com a versão 6.0 do Spyder 
!pip uninstall statsmodels
!pip install statsmodels

#%% Instalando os pacotes necessários 

pip install sklearn
pip install numpy
pip install pandas
pip install matplotlib
pip install imblearn

#%% Teste de Heterocedasticisdade e Multicolinearidade
# Teste realizado devido ao fato da acurácia do modelo aumentar de maneira 
#
inversamente proporcional ao tamanho da amostra
#%% Importando as bibliotecas
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

#%% Carregando o banco de dados
data = pd.read_excel('lista_saf.xlsx')

#%% Amostragem de 100% dos dados
df_sample = data.sample(frac=1, random_state=42)

#%% Separando as variáveis independentes e a variável dependente
X = df_sample[['CCO', 'NIVEL', 'LINHA', 'TX_TIPO_LOCALIDADE','TIPO_GRSIST_FALHA', 'LOCALIDADE_FALHA']]
y = df_sample['CANCELAMENTO_FALHA']

# Variavel  'TIPO_GRSIST_FALHA' apresenta VIF 9.137548 , ID_GRSIST_FALHA  9.321474
#   TIPO_GRSIST_FALHA  12.070084 100% DF
#     ID_GRSIST_FALHA  12.192567 100% DF 
#%% Convertendo variáveis categóricas usando LabelEncoder
le = LabelEncoder()
X = X.apply(le.fit_transform)

#%% Aplicando Random Over-Sampling para balanceamento das classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

#%% Verificando a Multicolinearidade com VIF
# Adicionando uma constante ao modelo para o intercepto
X_resampled_const = sm.add_constant(X_resampled)

# Calculando o VIF para cada variável
vif = pd.DataFrame()
vif["Variável"] = X.columns
vif["VIF"] = [variance_inflation_factor(X_resampled_const.values, i+1) for i in range(X_resampled_const.shape[1]-1)]

print(vif)

#%% Verificando a Heterocedasticidade com Teste de Breusch-Pagan
# Ajustando um modelo de regressão linear para os dados
model = sm.OLS(y_resampled, X_resampled_const).fit()

# Realizando o teste de Breusch-Pagan
test_bp = het_breuschpagan(model.resid, X_resampled_const)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, test_bp)))

#%% Plotando os resíduos para avaliar visualmente a heterocedasticidade
residuals = model.resid
fitted = model.fittedvalues

plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.title('Resíduos vs Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.show()
