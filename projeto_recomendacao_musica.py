# Databricks notebook source
# MAGIC %md
# MAGIC ## Funcionalidades e tratamentos realizados: 
# MAGIC Carregar dados no DBFS; 
# MAGIC Transformar o SQL DataFrame em Pandas DataFrame;
# MAGIC Adequar os tipos das colunas utilizando o astype do Pandas;
# MAGIC Utilizar o método replace do DataFrame do Pandas.

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/dados

# COMMAND ----------

caminho_data = "dbfs:/FileStore/dados/data.csv"
caminho_data_artist = "dbfs:/FileStore/dados/data_by_artist.csv"
caminho_data_genres = "dbfs:/FileStore/dados/data_by_genres.csv"
caminho_data_year = "dbfs:/FileStore/dados/data_by_year.csv"
caminho_data_w_genres = "dbfs:/FileStore/dados/data_w_genres.csv"

# COMMAND ----------

df_data = spark.read.csv(caminho_data, inferSchema=True, header=True)
display(df_data)

# COMMAND ----------

type(df_data)

# COMMAND ----------

df_data = df_data.pandas_api()

# COMMAND ----------

type(df_data)

# COMMAND ----------

df_data.head()

# COMMAND ----------

df_data.info()

# COMMAND ----------

colunas_float = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
colunas_int = ['duration_ms', 'mode', 'key', 'explicit', 'popularity']


# COMMAND ----------

df_data[colunas_float] = df_data[colunas_float].astype(float)
df_data[colunas_int] = df_data[colunas_int].astype(int)

# COMMAND ----------

df_data.info()

# COMMAND ----------

df_data.head()

# COMMAND ----------

type(df_data.artists.iloc[0])

# COMMAND ----------

x = df_data.iloc[0:9]

# COMMAND ----------

x['artists'] = x.artists.str.replace("\[|\]|\'", "")
x['artists'] = x.artists.str.replace(",", ";")

# COMMAND ----------

x

# COMMAND ----------

df_data['artists'] = df_data.artists.str.replace("\[|\]|\'", "")
df_data['artists'] = df_data.artists.str.replace(",", ";")

# COMMAND ----------

df_data.head()

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/dados_tratados

# COMMAND ----------

df_data.to_parquet('/FileStore/dados_tratados/data.parquet')

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/dados_tratados

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segunda parte de tratamentos

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

path = '/FileStore/dados_tratados/data_tratado.parquet/'
df_data = ps.read_parquet(path)
df_data.head()

# COMMAND ----------

df_data.describe()

# COMMAND ----------

len(df_data.year.unique())

# COMMAND ----------

df_data.year.value_counts()

# COMMAND ----------

df_data.year.value_counts().sort_index()

# COMMAND ----------

df_data.year.value_counts().sort_index().plot.bar()

# COMMAND ----------

df_data['decade'] = df_data.year.apply(lambda year: f'{(year//10)*10}s')
# (year//10)*10 divide por 10 e retira o ultimo digito do ano. depois multiplica por 10 para ter a decada. o f'{}'s é tratamendo de string padrão

# COMMAND ----------

df_data.head()

# COMMAND ----------

df_data_2 = df_data[['decade']] # para retornar no formato de dataframe ao invés de series, precisa de [[]]
df_data_2['qtd'] = 1

# COMMAND ----------

df_data_2 = df_data_2.groupby('decade').sum()
df_data_2


# COMMAND ----------

df_data_2.sort_index().plot.bar(y='qtd')

# COMMAND ----------

# DBTITLE 1,Impacto do ano de lançamento
# MAGIC %fs
# MAGIC ls /

# COMMAND ----------

df_data_year = spark.read.csv(caminho_data_year, inferSchema=True, header=True)
display(df_data_year)

# COMMAND ----------

df_data_year = df_data_year.pandas_api()
df_data_year.info()

# COMMAND ----------

df_data_year.to_parquet('/FileStore/dados_tratados/df_data_year.parquet')

# COMMAND ----------

path_data_year = '/FileStore/dados_tratados/df_data_year.parquet'
df_year = ps.read_parquet(path_data_year)
df_year.head()

# COMMAND ----------

len(df_year.year.unique())

# COMMAND ----------

df_year.plot.line(x='year', y='duration_ms')

# COMMAND ----------

# DBTITLE 1,Analise de algumas caracteristicas
df_year.plot.line(x='year', y=['acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'speechiness', 'valence'])

# COMMAND ----------

# DBTITLE 1,Como as características das músicas variam por década

# função apply() do Pandas e uma função lambda para transformar o ano em uma string com a década correspondente. 
# Add coluna "qtd" para contabilizar o número de músicas em cada década.

df_year['decade'] = df_year['year'].apply(lambda row: f'{(row//10)*10}s')
df_year['qtd'] = 1

# COMMAND ----------

# agregamos os dados das características das músicas utilizando o groupby por década e calculando a média de cada uma delas. 
# Para a coluna "qtd", usamos a função sum(), pois queremos saber o total de músicas em cada década. 
# Ordenamos o resultado pelo índice e redefinimos o índice do DataFrame.

df_year_2 = df_year.groupby('decade').agg({
    'acousticness': 'mean',
    'danceability': 'mean',
    'duration_ms': 'mean',
    'energy': 'mean',
    'instrumentalness': 'mean',
    'liveness': 'mean',
    'loudness': 'mean',
    'speechiness': 'mean', 
    'tempo': 'mean',
    'valence': 'mean', 
    'popularity': 'mean',
    'qtd': 'sum' }).sort_index().reset_index()

# COMMAND ----------

#criamos uma visualização em linha para analisar como as características das músicas variam por década. 
# Plotamos as colunas correspondentes às características, em relação à coluna "década".

df_year_2.plot.line(x='decade', y=['acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'speechiness', 'valence'])

#Há alguma característica que se destaca em uma determinada década?
#Existe alguma tendência geral nas características das músicas ao longo das décadas?
#A década de lançamento de uma música influencia em suas características?       
