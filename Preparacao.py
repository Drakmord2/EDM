#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata

# -----------------------------------------------------------------------------
# Globais
# -----------------------------------------------------------------------------

STR_FORMAT = '%Y-%m-%d %H:%M:%S'


# -----------------------------------------------------------------------------
# Metodos Auxiliares
# -----------------------------------------------------------------------------

def get_media_prova(row):
    n1 = str(row.PRIMEIRA_PROVA)
    n2 = str(row.SEGUNDA_PROVA)
    media = (float(n1) + float(n2)) / 2
    
    return media

def get_media_forum(row):
    n1 = str(row.FORUM01)
    n2 = str(row.FORUM02)
    n3 = str(row.FORUM03)
    n4 = str(row.FORUM04)
    
    notas = [n1, n2, n3, n4]
    notas = list(map(lambda x: float(x),notas))
    
    media = np.average(notas)
    
    return media

def rename_columns(df):
    df = df.rename(clean_string, axis='columns')
    return df

def clean_string(string):
    string = string.replace(' ', '_')
    string = str.upper(string)
    string = strip_accents(string)
    
    return string

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def get_date(ts):
    date = datetime.fromtimestamp(ts).strftime(STR_FORMAT)
    return date

def get_tempo_total(row):
    fim = datetime.strptime(row.DATA_DE_FINAL, STR_FORMAT)
    inicio = datetime.strptime(row.DATA_DE_INICIO, STR_FORMAT)
    
    tempo = fim - inicio
    
    return tempo

def get_subsets_semestre(df_base, path):
    semestres = []
    for ano in range(2009,2017,1):
        semestres.append(str(ano)+".1")
        semestres.append(str(ano)+".2")
    
    for semestre in semestres:
        df_curso = df_base.loc[df_base['SEMESTRE'] == semestre]
        if len(df_curso) != 0:
            df_curso.to_csv(path+semestre+'.csv', sep=';', decimal=',')


# -----------------------------------------------------------------------------
# Preparação
# -----------------------------------------------------------------------------

# Base
print("\n-Load")
df_base_original = pd.read_csv('./Base/baseGeral.csv', sep=';', decimal=',')

# Limpeza
print("-Limpeza")
df_base_original = rename_columns(df_base_original)
df_base = df_base_original.drop('VAR15', axis=1)

del df_base_original # Comentar se quiser comparar com base original

df_base['DATA_DE_INICIO'] = df_base['DATA_DE_INICIO'].apply(lambda ts: get_date(ts))
df_base['DATA_DE_FINAL'] = df_base['DATA_DE_FINAL'].apply(lambda ts: get_date(ts))

# Transformação
print("-Transformação")
df_base['MEDIA_CALCULADA_PROVA'] = df_base.apply(lambda row: get_media_prova(row), axis=1)
df_base['MEDIA_CALCULADA_FORUM'] = df_base.apply(lambda row: get_media_forum(row), axis=1)
df_base['TEMPO_DE_CURSO'] = df_base.apply(lambda row: get_tempo_total(row), axis=1)

# Seleção
print("-Seleção de subsets")
df_base = df_base.sort_values(
        by=['CURSO', 'SEMESTRE', 'PERIODO', 'NOME_DA_DISCIPLINA']
        )

prev = 0
maior = ""
cursos = df_base['CURSO'].unique()
for curso in cursos:
    tam_curso = len(df_base.loc[df_base['CURSO'] == curso])
    if tam_curso > prev:
        prev = tam_curso
        maior = curso

df_curso = df_base.loc[df_base['CURSO'] == maior]
get_subsets_semestre(df_curso, './Base/Subsets/base'+maior)

print("\n-Done")