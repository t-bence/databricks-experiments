# Databricks notebook source
# MAGIC %md
# MAGIC # Aim
# MAGIC
# MAGIC The aim of this notebook is to inspect the internals of a Spark Structured Streaming query. Look at the checkpoints and learn about how state management is done.

# COMMAND ----------

from datetime import datetime

# COMMAND ----------

BASE_PATH = "dbfs:/bence_streaming_experiment"

# COMMAND ----------

dbutils.fs.rm(BASE_PATH, True)

# COMMAND ----------

dbutils.fs.mkdirs(BASE_PATH)

CHECKPOINT_PATH = f"{BASE_PATH}/checkpoint"
dbutils.fs.mkdirs(CHECKPOINT_PATH)

FILES_PATH = f"{BASE_PATH}/streaming"
dbutils.fs.mkdirs(FILES_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC Erase checkpoint folder down here, optionally

# COMMAND ----------

#dbutils.fs.rm(CHECKPOINT_PATH, True)

# COMMAND ----------

class FileGenerator:
    def __init__(self, folder: str):
        self.folder = folder
        self.id = 1

    def write_file(self) -> int:
        import json
        import random
        import hashlib

        timestamp = datetime.now()

        # Generate a random integer ID
        random_count = random.randint(1, 10000)

        # Generate a random hash value
        random_hash = hashlib.sha256(str(random_count).encode()).hexdigest()

        # Create a JSON object
        data = {
            "id": self.id,
            "timestamp": timestamp.isoformat(),
            "count": random_count,
            "hash": random_hash
        }

        # Convert the JSON object to a string
        data_str = json.dumps(data)

        # Define the file path
        file_path = f"{self.folder}/{timestamp.strftime('%Y%m%d%H%M%S')}.json"

        # Write the JSON string to a file in DBFS
        dbutils.fs.put(file_path, data_str, overwrite=True)

        self.id += 1
        return self.id - 1 # the ID of the written file


# COMMAND ----------

generator = FileGenerator(FILES_PATH)

# COMMAND ----------

def show_files():
    files = dbutils.fs.ls(FILES_PATH)
    if files: 
        display(files)
    else:
        print("No files found.")

# COMMAND ----------

show_files()

# COMMAND ----------

generator.write_file()

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

query = (spark.readStream
    .schema("id INT, timestamp TIMESTAMP, count INT, hash STRING")
    .json(FILES_PATH)
    .groupBy("id")
    .agg(
        F.avg("count").alias("avg_count"),
        F.min("timestamp").alias("min_timestamp")
    )
    .writeStream
    .trigger(availableNow=True)
    .outputMode("complete")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .queryName("bence_test")
    .format("memory")
    .start()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's see the folders!
# MAGIC
# MAGIC **Note:** state folder only appears, when there is some aggregation (stateful transformation). commit, offsets and sources folders are there, along with metadata, even if there is no aggregation.

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ## Commits

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH + "/commits"))

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/commits/0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Offsets

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH + "/offsets"))

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/offsets/0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sources

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH + "/sources"))

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/sources/0/0

# COMMAND ----------

# MAGIC %md
# MAGIC ## State

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH + "/state"))

# COMMAND ----------

display(dbutils.fs.ls(CHECKPOINT_PATH + "/state/0/0"))

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/state/0/_metadata/metadata

# COMMAND ----------

# MAGIC %sh cat /dbfs/bence_streaming_experiment/checkpoint/state/0/0/1.delta

# COMMAND ----------

# MAGIC %md
# MAGIC This is a compressed file that I did not manage to read manually. But it turns out there is a built-in way to read the state store:
# MAGIC https://docs.databricks.com/en/structured-streaming/read-state.html
# MAGIC

# COMMAND ----------

df = (spark.read
  .format("statestore")
  .load(CHECKPOINT_PATH)
  .display()
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader state
# MAGIC
# MAGIC You can also query the state of Auto Loader with the following syntax -- but this is OS Spark file reading, so this is not applicable here.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT * FROM cloud_files_state('path/to/checkpoint');
# MAGIC

# COMMAND ----------


