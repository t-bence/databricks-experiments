# Databricks notebook source
# MAGIC %md
# MAGIC # Optuna framework test
# MAGIC
# MAGIC The aim of this notebook is to try out the Optuna optimization framework in Databricks env.
# MAGIC
# MAGIC Sample code is from https://optuna.org/#code_examples

# COMMAND ----------

# MAGIC %pip install optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sklearn
import optuna

# COMMAND ----------

import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 10000

# Generate random data for 6 numerical predictors
predictor_1 = np.random.randn(num_samples)
predictor_2 = np.random.randn(num_samples)
predictor_3 = np.random.randn(num_samples)
predictor_4 = np.random.randn(num_samples)
predictor_5 = np.random.randn(num_samples)
predictor_6 = np.random.randn(num_samples)

# Generate a random binary label (0 or 1)
label = np.random.randint(0, 2, num_samples)

# Create a DataFrame
df = pd.DataFrame({
    'predictor_1': predictor_1,
    'predictor_2': predictor_2,
    'predictor_3': predictor_3,
    'predictor_4': predictor_4,
    'predictor_5': predictor_5,
    'predictor_6': predictor_6,
    'label': label
})

# Display the first few rows of the DataFrame
print(df.head())

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {df.shape[0]} records in our source dataset")

# split target variable into it's own dataset
target_col = "label"
X_all = df.drop(labels=target_col, axis=1)
y_all = df[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.90, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# 1. Define an objective function to be maximized.
def objective(trial):

    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    n_estimators = trial.suggest_int('n_estimators', 2, 32, log=True)

    rf = sklearn.ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    
    accuracy = sklearn.metrics.accuracy_score(y_test, rf.fit(X_train, y_train).predict(X_test))
    return accuracy

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# COMMAND ----------


