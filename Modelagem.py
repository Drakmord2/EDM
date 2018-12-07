#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest

# -----------------------------------------------------------------------------
# Globais
# -----------------------------------------------------------------------------

STR_FORMAT = '%Y-%m-%d %H:%M:%S'


# -----------------------------------------------------------------------------
# Metodos Auxiliares
# -----------------------------------------------------------------------------

def get_date(ts):
    date = datetime.fromtimestamp(ts).strftime(STR_FORMAT)
    return date


def drop_invalid_columns(df):
    colsdrop = ['Unnamed: 0', 'VAR19', 'VAR29', 'VAR30']
    df = df.drop(colsdrop, axis=1)
    return df


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_correlations(df, n=5, asc=False, absolute=False):
    au_corr = df.corr().abs().unstack() if absolute else df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=asc)
    return au_corr[0:n]

def get_anomalies(base, reprovados=False):
    # Base de dado com e sem headers
    sample = base.copy()
    sample.columns = range(sample.shape[1])
    
    #Isolation Forest
    clf = IsolationForest(max_samples='auto', contamination=0.01, n_jobs=-1)
    clf.fit(sample)
    
    # Deteccao de anomalias
    scores = clf.decision_function(sample)
    predict  = clf.predict(sample)
    num_outliers = predict.tolist().count(-1)
    
    outliers = []
    outliers_position = []
    while len(outliers) < num_outliers:
        outliers.append(scores.min())
        
        outliers_position.append(scores.argmin()+1)
        scores = np.delete(scores, scores.argmin())
    
    # Anomalias encontradas    
    anomalias = base.iloc[outliers_position]
    anomalias = anomalias.sort_values(
            by=['DESEMPENHO'],
            ascending=False
            )
    anomalias = anomalias.drop_duplicates()
    # Correlacoes
    anomalias_aprovados = anomalias.loc[anomalias['DESEMPENHO_BINARIO']==0]
    corr_aprovados = get_top_correlations(anomalias_aprovados, 4000, absolute=True)
#    corr_aprovados = anomalias_aprovados.corr()
#    corr_aprovados = corr_aprovados['DESEMPENHO']
    
    if reprovados:
        anomalias_reprovados = anomalias.loc[anomalias['DESEMPENHO_BINARIO']==1]
        corr_reprovados = anomalias_reprovados.corr()
        corr_reprovados = corr_reprovados['DESEMPENHO']
        
        corr = pd.concat([corr_aprovados, corr_reprovados], axis=1)
        corr.columns = ["DESEMPENHO_APROVADOS", "DESEMPENHO_REPROVADOS"]
        
        return anomalias_aprovados, anomalias_reprovados, corr
        
    return anomalias_aprovados, corr_aprovados, None, None


def bar_plot_series(serie, name, save=False):
    plt.figure(figsize=(20,10))
    serie.plot.bar()
    if save:
        plt.savefig('./Outputs/Figuras/'+name+'.svg', bbox_inches='tight')
        return
    plt.show()

# -----------------------------------------------------------------------------
# Modelagem
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------")
# Base
print("\n- Carregar dados")
base = './Base/Subsets/basePedagogia.csv'
df_base = pd.read_csv(base, sep=';', decimal=',', index_col=False)
df_base = drop_invalid_columns(df_base)

# Seleção
print("- Seleção")
df_base = df_base.sort_values(
        by=['VAR31', 'VAR24', 'DESEMPENHO'],
        ascending=False
        )

print("- Analise")
print("  - Correlações")
# TODO Normalizar dados de entrada
# TODO Variar parametros do isolation forest
# TODO Explicar variaveis de interesse
# TODO Escrever artigo detalhando cada fase da CRISP-DM

# Variaveis de interesse (vide 'variaveis_tese.xslx')
df_corr = df_base[['VAR02', 'VAR03', 'VAR04','VAR06', 'VAR07', 'VAR16','VAR18', 'VAR20',
                   'VAR24', 'VAR31', 'VAR33', 'VAR34', 'MEDIA_PROVAS', 'MEDIA_FORUM', 
                   'MEDIA_WEBQUEST', 'DESEMPENHO', 'DESEMPENHO_BINARIO']]

df_corr = df_corr.loc[df_corr['DESEMPENHO_BINARIO']==0]
#correlacoes_positivas = get_top_correlations(df_corr, 60)
#correlacoes_negativas = get_top_correlations(df_corr, 15, asc=True)
correlacoes_absolutas = get_top_correlations(df_corr, 4000, absolute=True)

print("  - Anomalias")

anomalias_aprovados, correlacoes_aprovados, _, _ = get_anomalies(df_corr)
diff = pd.concat([correlacoes_absolutas, correlacoes_aprovados], axis=1)
diff = diff.rename(index=str, columns={0:"Base", 1:"Anomalias"})

print("\n--------------------------------------------------------------------")
