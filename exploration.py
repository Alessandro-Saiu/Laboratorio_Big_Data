import os
import sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class Evaluator(object):
    def __init__(self, dataframe=''):
        self.spark = SparkSession.builder.appName('Exploration').getOrCreate()
        self.df = self.spark.read.csv(dataframe, inferSchema=True, header=True)

    def select_columns(self, columns):
        self.df = self.df.select(columns)

    def start_expl(self, categorical=[]):
        self.df.printSchema()
        for col in self.df.columns:
            self.df.describe([col]).show()
            print(col, "\t", "with null values: ", self.df.filter(self.df[col].isNull()).count())
            print(col, "\t", "with NaN values: ", self.df.filter(self.df[col] == "Nan").count())
        if categorical:
            for i in categorical:
                print("\n", self.df.select(i).distinct().rdd.map(lambda r: r[0]).collect())
