# Databricks notebook source
# MAGIC %fs
# MAGIC ls /FileStore/dados

# COMMAND ----------

caminho_data = "dbfs:/FileStore/dados/data.csv"

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
