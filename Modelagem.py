#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

# Base
print("\n- Carregar dados")
base = './Base/Subsets/basePedagogia.csv'
df_base = pd.read_csv(base, sep=';', decimal=',', index_col=False)
df_base = drop_invalid_columns(df_base)

# Variaveis de interesse (vide 'variaveis_tese')
# Var02-10; Var24; Var31-c; 

variaveis = {"VAR24": "MEDIA_ACESSO_SEMANAL", "VAR31": "TOTAL_ACESSOS"}

# Seleção
print("- Seleção")
df_base = df_base.sort_values(
        by=['VAR31', 'VAR24', 'DESEMPENHO'],
        ascending=False
        )
df_base = df_base.rename(variaveis, axis=1)

print("- Analise")
colsdrop = ['ID_DO_ALUNO', 'CURSO','DATA_DE_INICIO','DATA_DE_FINAL', 
            'SEMESTRE', 'NOME_DA_DISCIPLINA','TEMPO_DE_CURSO']
df_corr = df_base.drop(colsdrop, axis=1)

correlacoes_positivas = get_top_correlations(df_corr, 60)
correlacoes_negativas = get_top_correlations(df_corr, 15, asc=True)
correlacoes_absolutas = get_top_correlations(df_corr, 60, absolute=True)

print("\n-Done")
