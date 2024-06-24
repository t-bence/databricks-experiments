# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow pipeline logging experiment
# MAGIC
# MAGIC This notebook is used to demonstrate how you can log a complete Spark pipeline with mlflow, just like you would log a model.
# MAGIC
# MAGIC The notebook was developed with DBR 14.3 LTS ML.

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Now comes the data and modelling part, taken from: https://spark.apache.org/docs/latest/ml-pipeline.html

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print(
        "(%d, %s) --> prob=%s, prediction=%f" % (
            rid, text, str(prob), prediction   # type: ignore
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Then let's do the mlflow part

# COMMAND ----------

import mlflow.spark

with mlflow.start_run(run_name="spark-pipeline-test") as run:
  mlflow.spark.log_model(model, "model")
  mlflow.log_param("maxIter", 10)

# COMMAND ----------

# MAGIC %md
# MAGIC Load model and do inference on dummy data

# COMMAND ----------

logged_model = 'runs:/33b6cdf562b34ba0bdc8fea4d1abb8b4/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = ([['spark i j k'], ['l m n'], ['spark hadoop spark'], ['apache hadoop']])

df = pd.DataFrame(data)
df.columns = ['text']

loaded_model.predict(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Others 
# MAGIC
# MAGIC - Pipeline class in pyspark repo: https://github.com/apache/spark/blob/master/python/pyspark/ml/pipeline.py
# MAGIC - mlflow.spark docs: https://mlflow.org/docs/latest/python_api/mlflow.spark.html
