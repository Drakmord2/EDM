#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# -----------------------------------------------------------------------------
# Modelagem
# -----------------------------------------------------------------------------

# Base
print("\n- Carregar dados")
base = './Base/Subsets/basePedagogia.csv'
df_base = pd.read_csv(base, sep=';', decimal=',', index_col=False)

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
colsdrop = ['Unnamed: 0','ID_DO_ALUNO', 'CURSO','DATA_DE_INICIO',
            'DATA_DE_FINAL', 'SEMESTRE', 'NOME_DA_DISCIPLINA','TEMPO_DE_CURSO',
            'VAR19', 'VAR29', 'VAR30']
df_corr = df_base.drop(colsdrop, axis=1)

correlacoes_altas = get_top_abs_correlations(df_corr, 70)

print("\n-Done")
