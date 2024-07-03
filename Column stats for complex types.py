# Databricks notebook source
# MAGIC %md
# MAGIC Check column stats in the Delta log for complex types:
# MAGIC - timestamp
# MAGIC - structs
# MAGIC - arrays
# MAGIC - strings

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a DF that contains these data types

# COMMAND ----------

from pyspark.sql.types import IntegerType, StructType, StructField, StringType, TimestampType, ArrayType
from pyspark.sql import SparkSession
from datetime import datetime

# Define the schema for the DataFrame
schema = StructType([
    StructField("int_col", IntegerType(), nullable=False),
    StructField("string_col", StringType(), nullable=False),
    StructField("timestamp_col", TimestampType(), nullable=False),
    StructField("struct_col", StructType([
        StructField("field1", StringType(), nullable=False),
        StructField("field2", StringType(), nullable=False)
    ])),
    StructField("array_col", ArrayType(StringType()), nullable=False)
])

# Create a list of data
data = [
    (1, "A", datetime(2024, 7, 3, 12, 0, 0), {"field1": "a", "field2": "b"}, ["item1", "item2", "item3"]),
    (2, "B", datetime(2024, 7, 4, 12, 0, 0), {"field1": "b", "field2": "a"}, ["item2", "item3", "item4"]),
]

# Create the DataFrame
df = spark.createDataFrame(data, schema)

# Show the DataFrame
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Save the DF to DBFS

# COMMAND ----------

path = "/dbfs/tmp/bence_toth_experiments"

# COMMAND ----------

dbutils.fs.rm(path, True)
df.coalesce(1).write.format("delta").mode("overwrite").save(path)

# COMMAND ----------

display(
  dbutils.fs.ls(path)
)

# COMMAND ----------

display(
  dbutils.fs.ls(path + "/_delta_log")
)

# COMMAND ----------

stats_df = spark.read.json(path + "/_delta_log/00000000000000000000.json")

stats_df.display()
json_rows = stats_df.collect()

# COMMAND ----------

stats = json_rows[-1]
contents = stats["add"]["stats"]
print(contents)

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC
# MAGIC ## Integer column
# MAGIC Works as expected
# MAGIC
# MAGIC ## String column
# MAGIC Works as expected
# MAGIC
# MAGIC ## Timestamp column
# MAGIC Works as expected
# MAGIC
# MAGIC ## Struct column
# MAGIC There is a min and a max entry for every field in the struct column.
# MAGIC
# MAGIC ## Array column
# MAGIC Only the is null information is logged, there is no min or max logged.

# COMMAND ----------


