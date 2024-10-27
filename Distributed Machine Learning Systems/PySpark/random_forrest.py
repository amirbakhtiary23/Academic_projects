#importing required libs
import time
r=time.time()
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as functions
import pyspark.ml.classification as cls
import pyspark.ml.feature as fe
import pyspark.ml.linalg as linalg
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as evaluator
#creating a spark session
spark_session=SparkSession.builder.appName("Bakhtiary_Q3_lr").getOrCreate()
spark_session.sparkContext.setLogLevel("ERROR")
#reading the csv file from hdfs 
dataset=spark_session.read.csv("hdfs://raspberrypi-dml0:9000/bakhtiary_810101114/heart.csv",inferSchema=True,header=True)
print(dataset.show())
stats1=dataset.select(
        functions.mean(functions.col("age")).alias("mean"),
        functions.stddev(functions.col("age")).alias("std"))

stats2=dataset.select(functions.mean(functions.col("trtbps")).alias("mean"),
        functions.stddev(functions.col("trtbps")).alias("std"))
stats3=dataset.select(functions.mean(functions.col("chol")).alias("mean"),
        functions.stddev(functions.col("chol")).alias("std"))
stats1=stats1.collect()
stats2=stats2.collect()
stats3=stats3.collect()
print("age mean:",stats1[0]["mean"],"std :",stats1[0]["std"])
print ("trtbps mean:",stats2[0]["mean"],"std:",stats2[0]["std"])
print ("chol mean:",stats3[0]["mean"],"std:",stats3[0]["std"])
train,test=dataset.randomSplit(weights=[0.85,0.15],seed=42)
print (train.count(),len(train.columns),"trainSet shape")
print (test.count(),len(test.columns),"testSet shape")
print ("columns :",train.columns)
features=train.columns
features.pop()

assembler=fe.VectorAssembler (inputCols=features,outputCol="features")
#dataset=assembler.transform(dataset)

lr=cls.RandomForestClassifier(maxDepth=20,featuresCol="features",labelCol="output")
#print ("coefs : ",lr.coefficients)
pipe=Pipeline(stages=[assembler,lr])

trained_model=pipe.fit(train)
#trained_model.stages[-1].setThreshold(0.5)
#print ("coefs:",trained_model.stages[-1].coefficients)
predictions=trained_model.transform(test)
print (predictions.show())
#y_pred=predictions.select("probability")
#y_true=test.select("output")
#print (y_pred.count(),y_true.count())
pred=predictions.select("prediction","output")

print (pred.show())

#print ("Model Accuracy",trained_model.summary.accuracy)
#print ("test accuracy",predictions.accuracy)
evaluate=evaluator()
evaluate.setPredictionCol("prediction")
evaluate.setLabelCol("output")
print ("acc",evaluate.evaluate(pred,{evaluate.metricName:"accuracy"}))
print ("precision",evaluate.evaluate(pred,{evaluate.metricName:"weightedPrecision"}))
print ("recall",evaluate.evaluate(pred,{evaluate.metricName:"weightedRecall"}))
print ("f1",evaluate.evaluate(pred,{evaluate.metricName:"f1"}))
print (f"took {r-time.time()}s")
#print ("precision :",evaluate.evaluate(predictions,{evaluate.metricName:"recall"}))"""
