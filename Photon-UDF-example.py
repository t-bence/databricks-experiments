# Databricks notebook source
# MAGIC %md
# MAGIC # Photon vs UDFs
# MAGIC
# MAGIC The docs say that Photon is not compatible with UDFs. https://learn.microsoft.com/en-us/azure/databricks/compute/photon#limitations I want to see what that looks like in practice: the query will not eve start using Photon, if there is a UDF in the middle, or it will use Photon before and after the UDF, just not during?
# MAGIC
# MAGIC Make sure this demo is ran on a cluster with Photon enabled :)
# MAGIC
# MAGIC Let's create a DataFrame, do transformations on it, ensure it uses Photon, then run it again with a UDF and see what happens!

# COMMAND ----------

import pyspark.sql.functions as F

num_rows = 10_000_000

df = (spark
  .range(num_rows)
  .toDF("id")
  .withColumn("value", F.rand())
)

df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC Do some transformations to ensure that Photon is used when there is no UDF...

# COMMAND ----------

(df.filter(F.col("value") > 0.5).count())

# COMMAND ----------

# MAGIC %md
# MAGIC This used Photon according to the Spark UI.
# MAGIC
# MAGIC Let's define a UDF

# COMMAND ----------

@F.udf("float")
def dummy_udf(input: float) -> float:
  return input

# COMMAND ----------

second = (df
  .withColumn("new_value", dummy_udf("value"))
  .filter(F.col("new_value") > 0.5)
)
second.count()

# COMMAND ----------

second.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC With this, we had Photon running until the UDF, then it was regular Spark.

# COMMAND ----------


