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


# -----------------------------------------------------------------------------
# Modelagem
# -----------------------------------------------------------------------------

# Base
print("\n-Carregar dados")
base = './Base/Subsets/basePedagogia.csv'
df_base = pd.read_csv(base, sep=';', decimal=',', index_col=False)

# Variaveis de interesse (vide 'variaveis_tese')
# Var02-10; Var24; Var31-c; 

variaveis = {"VAR24": "MEDIA_ACESSO_SEMANAL", "VAR31": "TOTAL_ACESSOS"}

# Seleção
print("-Seleção")
cols = ['SEMESTRE','NOME_DA_DISCIPLINA','PERIODO','ID_DO_ALUNO','VAR24', 
        'VAR31', 'DESEMPENHO', 'DESEMPENHO_BINARIO']
df_base = df_base[cols]
df_base = df_base.sort_values(
        by=['VAR31', 'VAR24', 'DESEMPENHO'],
        ascending=False
        )
#df_base = df_base[:200]
df_base = df_base.rename(variaveis, axis=1)

df_corr = df_base.drop('ID_DO_ALUNO', axis=1)
correlacoes = df_corr.corr()

print("\n-Done")
