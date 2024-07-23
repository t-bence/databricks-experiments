# Databricks notebook source
import dlt
import pyspark.sql.functions as F

num_rows = 100

@dlt.table
def raw():
    return (spark
        .range(num_rows)
        .toDF("id")
        .withColumn("random_float", F.rand())
    )

# COMMAND ----------

from package.code import twice

# COMMAND ----------

@dlt.table
def silver():
    return (dlt
        .read("raw")
        .withColumn("twice", twice(F.col("random_float")))
    )
