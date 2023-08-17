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

# COMMAND ----------

# MAGIC %md
# MAGIC #Terceira parte -  perguntas
# MAGIC * Quais são os 10 top artistas ?
# MAGIC * Qual o gênero musical dos 10 top artistas ?
# MAGIC * Quais sao os 10 top artistas ?
# MAGIC * Quais artistas estão nos 10 top gêneros ?
# MAGIC

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /

# COMMAND ----------

caminho_data_artist = "dbfs:/FileStore/dados/data_by_artist.csv"
caminho_data_genres = "dbfs:/FileStore/dados/data_by_genres.csv"
caminho_data_w_genres = "dbfs:/FileStore/dados/data_w_genres.csv"

# COMMAND ----------

df_data_artist = spark.read.csv(caminho_data_artist, inferSchema=True, header=True)
df_data_genres = spark.read.csv(caminho_data_genres, inferSchema=True, header=True)
df_data_w_genres = spark.read.csv(caminho_data_w_genres, inferSchema=True, header=True)

# COMMAND ----------

df_data_artist.printSchema()

# COMMAND ----------

df_data_genres.printSchema()

# COMMAND ----------

df_data_w_genres.printSchema()

# COMMAND ----------

df_data_artist = df_data_artist.pandas_api()
df_data_genres = df_data_genres.pandas_api()
df_data_w_genres = df_data_w_genres.pandas_api()

# COMMAND ----------


df_data_artist.to_parquet('/FileStore/dados_tratados/data_artist.parquet')
df_data_genres.to_parquet('/FileStore/dados_tratados/data_genres.parquet')
df_data_w_genres.to_parquet('/FileStore/dados_tratados/data_w_genres.parquet')

# COMMAND ----------

path_data_artist = '/FileStore/dados_tratados/data_artist.parquet'
path_data_genres = '/FileStore/dados_tratados/data_genres.parquet'
path_data_w_genres = '/FileStore/dados_tratados/data_w_genres.parquet'


# COMMAND ----------

df_data_artist = ps.read_parquet(path_data_artist)
df_data_genres = ps.read_parquet(path_data_genres)
df_data_w_genres = ps.read_parquet(path_data_w_genres)


# COMMAND ----------

df_data_artist['artists'] = df_data_artist.artists.str.replace("\"", "")

# COMMAND ----------

df_data_artist.head()

# COMMAND ----------

artistas_ordenados = df_data_artist.sort_values(by='count', ascending=False)
artistas_ordenados.head()

# COMMAND ----------

top_artistas = artistas_ordenados.iloc[0:10]
top_artistas

# COMMAND ----------

plot_title = 'Top 10 Artistas'
top_artistas.plot.bar(x='count', y = 'artists', title = plot_title)

# COMMAND ----------

# DBTITLE 1,Genero musical dos top 10 artistas
lista_artistas = top_artistas.artists.unique().to_list()
lista_artistas

# COMMAND ----------

df_data_genres.head()

# COMMAND ----------

df_data_w_genres['artists'] = df_data_w_genres.artists.str.replace("\"", "")

# COMMAND ----------

df_data_w_genres.head()

# COMMAND ----------

artistas_genero = df_data_w_genres.loc[df_data_w_genres['artists'].isin(lista_artistas)]
artistas_genero = artistas_genero[['genres', 'artists']]
display(artistas_genero)

# COMMAND ----------

# DBTITLE 1,Quais são os 10 principais gêneros musicais?
df_data_w_genres['qtd'] = 1  # Cria uma coluna de quantidade de 1 para cada linha
df_data_w_genres_2 = df_data_w_genres[['genres', 'qtd']]  # Seleciona apenas as colunas de gênero e quantidade
df_data_w_genres_2_ordenado = df_data_w_genres_2.groupby('genres').sum().sort_values(by='qtd', ascending=False).reset_index() # Agrupa os dados por gênero e soma a quantidade de músicas de cada gênero, em seguida ordena os resultados em ordem decrescente
top_generos = df_data_w_genres_2_ordenado.loc[0:10] # Seleciona os 10 gêneros mais populares

plot_title = 'Top 10 Gêneros'
top_generos.plot.bar(x='qtd', y='genres', title=plot_title) # Plota um gráfico de barras para os top 10 gêneros

top_generos_2 =df_data_w_genres_2_ordenado.loc[1:11] # Seleciona os gêneros do 2º ao 11º mais populares

plot_title = 'Top 10 Gêneros'
top_generos_2.plot.bar(x='qtd', y='genres', title=plot_title) # Plota um gráfico de barras para os gêneros do 2º ao 11º mais populares

