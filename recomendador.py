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

# COMMAND ----------

# MAGIC %md
# MAGIC # PCA
# MAGIC * PCA (Principal Component Analysis) é uma técnica estatística utilizada para redução de dimensionalidade e análise de dados. É uma das técnicas mais comuns de análise multivariada e é amplamente utilizada em várias áreas, como ciência de dados, aprendizado de máquina e reconhecimento de padrões.
# MAGIC * O objetivo principal do PCA é transformar um conjunto de variáveis correlacionadas em um novo conjunto de variáveis não correlacionadas, chamadas de componentes principais. Esses componentes principais são combinações lineares das variáveis originais e são ordenados de forma que o primeiro componente principal explique a maior parte da variabilidade dos dados, o segundo componente principal explique a maior parte da variabilidade restante, e assim por diante.
# MAGIC * A redução de dimensionalidade ocorre ao descartar os componentes principais de menor importância, o que permite representar os dados originais em um espaço de menor dimensão. Isso pode ser útil quando há um grande número de variáveis e se deseja simplificar a análise ou visualização dos dados.
# MAGIC * Também pode ser usado para identificar padrões e tendências nos dados, detectar outliers e ruídos, e realizar tarefas como compressão de dados e reconstrução de dados faltantes.
# MAGIC

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

pca_pipeline = Pipeline(stages=[VectorAssembler(inputCols=X, outputCol='features'),
                                StandardScaler(inputCol='features', outputCol='features_scaled'),
                                PCA(k=6, inputCol='features_scaled', outputCol='pca_features')])

# COMMAND ----------

model_pca_pipeline = pca_pipeline.fit(df_data)

# COMMAND ----------

projection = model_pca_pipeline.transform(df_data)

# COMMAND ----------

projection.select('pca_features').show(truncate=False, n=5)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

SEED = 1224

# COMMAND ----------

kmeans = KMeans(k=50, featuresCol='pca_features', predictionCol='cluster_pca', seed=SEED)

# COMMAND ----------

modelo_kmeans = kmeans.fit(projection)

# COMMAND ----------

projection_kmeans = modelo_kmeans.transform(projection) 

# COMMAND ----------

projection_kmeans.select(['pca_features','cluster_pca']).show()

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

# COMMAND ----------

projection_kmeans = projection_kmeans.withColumn('x', vector_to_array('pca_features')[0])\
                                   .withColumn('y', vector_to_array('pca_features')[1])

# COMMAND ----------

projection_kmeans.select(['x', 'y', 'cluster_pca', 'artists_song']).show()

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

fig = px.scatter(projection_kmeans.toPandas(), x='x', y='y', color='cluster_pca', hover_data=['artists_song'])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Features still here:
# MAGIC * Vetorizar seus dados com VectorAssembler;
# MAGIC * Padronizar os dados com StandardScaler;
# MAGIC * Reduzir a dimensão dos dados com PCA;
# MAGIC * Criar clusters com K-Means;
# MAGIC * Analisar os agrupamentos.

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'

# COMMAND ----------

type(projection_kmeans)

# COMMAND ----------

projection_kmeans = projection_kmeans

# COMMAND ----------

cluster = projection_kmeans.filter(projection_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
cluster

# COMMAND ----------

musicas_recomendadas = projection_kmeans.filter(projection_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
musicas_recomendadas.show()

# COMMAND ----------

#Para calcular as distâncias entre as músicas recomendadas e a nossa música de referência, precisamos obter as componentes da música de referência. 
#No início, fizemos a extração do cluster dessa música de referência.
#Agora, queremos buscar os valores que descrevem essa música

# COMMAND ----------

componenetes_musica = musicas_recomendadas \
                        .filter(musicas_recomendadas.artists_song == nome_musica)\
                        .select('pca_features').collect()[0][0]
componenetes_musica                                                        

# COMMAND ----------

#Com esses dados, podemos calcular a distância entre a pca_features das outras músicas do cluster.
#Existem várias maneiras de calcular a distância, pode ser a distância euclidiana, ou a distância de Manhattan
#O cálculo pode ser simples, quando estivermos considerando, por exemplo, duas dimensões. 
# Calcular a distância entre dois pontos no plano XY pode ser simples. Mas em uma dimensão de 6, a conta vai ser mais complexa. Para isso, usaremos a biblioteca SciPy

# COMMAND ----------

from scipy.spatial.distance import euclidean
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f

# COMMAND ----------

#Precisamos percorrer toda a lista de músicas e calcular a distância para cada uma delas, salvando os resultados em uma coluna "distancia".
#Então, os dados não estão concentrados em um só node, estão em diversos nodes. Precisaremos usar outras ferramentas para fazer esse apply.
#Precisamos transformá-la em uma função compatível com o Spark


# COMMAND ----------

def calcula_distance(value):
    return euclidean(componenetes_musica, value)

#transformamos nossa função calcula_distance para em função Spark.
udf_calcula_distance = f.udf(calcula_distance, FloatType())

musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

# COMMAND ----------

recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])
recomendadas.show(truncate=False)

# COMMAND ----------

recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

recomendadas.show()
