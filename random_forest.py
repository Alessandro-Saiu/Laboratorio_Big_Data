import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import feature
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class Preparation(object):
    def __init__(self, dataframe=''):
        self.spark = SparkSession.builder.appName('RF').getOrCreate()
        self.df = self.spark.read.csv(dataframe, inferSchema=True, header=True)

    def select_columns(self, columns):
        self.df = self.df.select(columns)
        return self.df

    def make_string_indexing(self, categorical):
        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(self.df)
                    for column in categorical]
        pipeline = Pipeline(stages=indexers)
        self.df = pipeline.fit(self.df).transform(self.df)
        self.df = self.df.drop(*categorical)
        self.df.show(5)
        return self.df

    def make_one_hot_encoding(self):
        index_list = [col for col in self.df.columns if col.find('_index') != -1]
        dummy_list = [col.replace('index', 'dummy') for col in index_list]
        encoder = feature.OneHotEncoder(inputCols=index_list,
                                        outputCols=dummy_list,
                                        dropLast=True)
        self.df = encoder.fit(self.df).transform(self.df)
        self.df = self.df.withColumn('winner', F.when(self.df.winner == 'white', 0).
                                     when(self.df.winner == 'black', 1).otherwise(2))
        self.df = self.df.drop(*index_list)
        self.df.show(5)
        return self.df

    def vectorize(self, target):
        features = self.df.drop(target).columns
        vector = VectorAssembler(inputCols=features, outputCol='features')
        self.df = vector.transform(self.df)
        self.df.show(5)
        return self.df, features


class RandomTree(object):
    def __init__(self, dataframe, features):
        self.spark = SparkSession.builder.appName('RT').getOrCreate()
        self.df = dataframe
        self.features = features

    def apply_model(self, target):
        train, test = self.df.randomSplit([0.8, 0.2], seed=87)
        forest_model = RandomForestClassifier(featuresCol='features',
                                              labelCol=target,
                                              predictionCol='prediction',
                                              maxDepth=9,
                                              numTrees=86,
                                              impurity='gini',
                                              subsamplingRate=.5).fit(train)
        predictions = forest_model.transform(test).select(target, 'prediction')
        predictions.head(10)

        evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=target)
        feature_weight = sorted(list(zip(forest_model.featureImportances, self.features)), reverse=True)
        print('F1 Score:', evaluator.evaluate(predictions, {evaluator.metricName: 'f1'}))
        print('Accuracy:', evaluator.evaluate(predictions, {evaluator.metricName: 'accuracy'}))
        print(feature_weight)

