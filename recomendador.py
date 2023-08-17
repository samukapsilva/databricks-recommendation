# Databricks notebook source
# MAGIC %fs
# MAGIC ls /FileStore/dados_tratados/

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

path = "/FileStore/dados_tratados/data_tratado.parquet/"
df_data = ps.read_parquet(path)

# COMMAND ----------

df_data.info()

# COMMAND ----------

df_data = df_data.dropna()

# COMMAND ----------

df_data.info()

# COMMAND ----------

df_data['artists_song'] = df_data.artists + ' - ' + df_data.name

# COMMAND ----------

df_data.head()

# COMMAND ----------

df_data.info()

# COMMAND ----------

X = df_data.columns.to_list()
X.remove('artists')
X.remove('id')
X.remove('name')
X.remove('artists_song')
X.remove('release_date')
X

# COMMAND ----------

# DBTITLE 1,DataFrame Transforming 
# pandas to spark
df_data = df_data.to_spark()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler


# COMMAND ----------

#A primeira coisa a fazer para utilizar esses dados no modelo de Machine Learning é vetorizar nossos dados.
# Isso porque temos diversas colunas que representam nossos dados, focando nas numéricas, e o MLlib espera que essas informações estejam todas contidas em uma coluna só, e representadas # através de uma lista. Isso tem a ver com a performance, com como ele funciona internamente.

#Para fazer essa transformação, temos uma ferramenta: o VectorAssembler. 
# #Vamos aplicá-la e criar uma nova coluna que tenha apenas essa informação

# COMMAND ----------

dados_encoded_vector = VectorAssembler(inputCols=X, outputCol='features').transform(df_data)

# COMMAND ----------

dados_encoded_vector.select('features').show(truncate=False, n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC #Vectorization
# MAGIC
# MAGIC Em resumo, o VectorAssembler é uma ferramenta importante no PySpark para transformar várias colunas em um único vetor, o que é útil para treinar modelos de aprendizado de máquina. Além disso, as matrizes densas e esparsas, representadas pelos objetos DenseVector e SparseVector, respectivamente, são importantes para armazenar informações de maneira eficiente, dependendo do conjunto de dados utilizado.
# MAGIC No caso do PySpark, o vetor é representado como um objeto do tipo DenseVector ou SparseVector. A diferença entre as duas matrizes é que a primeira representa uma matriz densa, ou seja, todos os elementos são armazenados, enquanto a segunda representa uma matriz esparsa, na qual apenas os elementos diferentes de zero são armazenados

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
model_scaler = scaler.fit(dados_encoded_vector)
dados_musicas_scaler = model_scaler.transform(dados_encoded_vector)

# COMMAND ----------

dados_musicas_scaler.select('features_scaled').show(truncate=False, n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC # StandardScale
# MAGIC A padronização dimensiona cada variável de entrada separadamente, subtraindo a média (chamada de centralização) e dividindo pelo desvio padrão para deslocar a distribuição a fim de ter uma média de 0 e um desvio padrão de 1. Esse processo pode ser pensado como subtrair o valor médio ou centralizar os dados e requer conhecer os valores de média e desvio padrão dos dados.
# MAGIC
# MAGIC Formula:
# MAGIC y = x-media/desvio padrão
# MAGIC Onde a media pode ser calculada como:
# MAGIC media = soma(x)/contagem(x)
# MAGIC Onde o desvio padrão pode ser calculado como:
# MAGIC desvio padrão = _______________
# MAGIC                V soma((x -media)²) / contagem(x)

# COMMAND ----------

k = len(X)
k

# COMMAND ----------

# Com nossas features escaladas, precisamos reduzir o nosso número de features
# Podemos utilizar a técnica do PCA (Principal Component Analysis - Análise dos Componentes Principais).

from pyspark.ml.feature import PCA

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

model_pca.explainedVariance

# COMMAND ----------

sum(model_pca.explainedVariance) * 100

# COMMAND ----------

lista_valores = [sum(model_pca.explainedVariance[0:i+1]) 
                 for i in range(k)]
lista_valores

# COMMAND ----------

import numpy as np

# COMMAND ----------

k = sum(np.array(lista_valores) <= 0.7)
k

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca_final = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

dados_musicas_pca_final.select('pca_features').show(truncate=False, n=5)
