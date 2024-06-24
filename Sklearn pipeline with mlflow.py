# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow pipeline logging experiment
# MAGIC
# MAGIC This notebook is used to demonstrate how you can log a complete sklearn pipeline with mlflow, just like you would log a model.
# MAGIC
# MAGIC The notebook was developed with DBR 14.3 LTS ML.

# COMMAND ----------

# MAGIC %md
# MAGIC Now comes the data and modelling part, taken from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def true_fun(X):
    return np.cos(1.5 * np.pi * X)


np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    )

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(
        "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()
        )
    )
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Then let's do the mlflow part

# COMMAND ----------

import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="sklearn-pipeline-test") as run:
  mlflow.sklearn.log_model(pipeline, "model")
  mlflow.log_metric("mse", scores.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Load model and do inference on dummy data

# COMMAND ----------

import mlflow
logged_model = 'runs:/f9b7cac09b374231b6a8c63f86ef622f/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = (0.1, )

loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------


