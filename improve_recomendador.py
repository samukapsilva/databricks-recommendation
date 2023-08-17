# Databricks notebook source
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Carregando dados de exemplo
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import rand

data = spark.createDataFrame([(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
                              (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)], ['features']).orderBy(rand())

# Criando o modelo K-means
kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(data)

# Avaliando a qualidade dos clusters com o SilhouetteEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(model.transform(data))
print(f"Silhouette score: {silhouette}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Carregando dados de exemplo
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import rand
data = spark.createDataFrame([(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
                              (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)], ['features']).orderBy(rand())

# Avaliando o modelo K-means com diferentes valores de k
silhouette_scores = []
k_values = range(2, 6)
for k in k_values:
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(data)
    evaluator = ClusteringEvaluator()
    silhouette_scores.append(evaluator.evaluate(model.transform(data)))

# Plotando o gráfico de silhueta vs número de clusters
plt.plot(k_values, silhouette_scores, '-o')
plt.xlabel('Número de Clusters')
plt.ylabel('Score de Silhueta')
plt.show()
