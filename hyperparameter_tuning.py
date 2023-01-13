import os
import sys
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from sklearn.model_selection import train_test_split

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class HyperParameters(object):
    def __init__(self, dataframe):
        self.spark = SparkSession.builder.appName('HP').getOrCreate()
        self.df = dataframe

    outSchema = StructType(
        [StructField('replication_id', IntegerType(), True), StructField('Accuracy', DoubleType(), True),
         StructField('num_trees', IntegerType(), True), StructField('depth', IntegerType(), True),
         StructField('criterion', StringType(), True)])

    @f.pandas_udf(outSchema, f.PandasUDFType.GROUPED_MAP)
    def run_model(self, pdf):
        num_trees = random.choice(list(range(50, 500)))
        depth = random.choice(list(range(2, 10)))
        criterion = random.choice(['gini', 'entropy'])
        replication_id = pdf.replication_id.values[0]
        X = pdf[['turns', 'white_rating', 'black_rating', 'victory_status_index', 'opening_eco_index']]
        y = pdf['winner']
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.33, random_state=42)
        clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth, criterion=criterion)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(clf.predict(X_cv), y_cv)
        res = pd.DataFrame({'replication_id': replication_id, 'Accuracy': accuracy, 'num_trees': num_trees,
                            'depth': depth, 'criterion': criterion}, index=[0])
        return res

    def tuning(self):
        replication_df = self.spark.createDataFrame(pd.DataFrame(list(range(1, 1001)), columns=['replication_id']))
        replicated_train_df = self.df.crossJoin(replication_df)
        results = replicated_train_df.groupby("replication_id").apply(self.run_model)
        results.sort(f.desc("Accuracy")).show()
