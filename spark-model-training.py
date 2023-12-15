from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.evaluation import RegressionEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("Sentiment_Analysis").getOrCreate()


# Train data
train_data = spark.read.csv(
    "hdfs:///user/maria_dev/train.csv", header="true", sep="\t", inferSchema="true"
)
test_data = spark.read.csv(
    "hdfs:///user/maria_dev/test.csv", header="true", sep="\t", inferSchema="true"
)

# 문자열을 리스트 형식으로 변환
list_converter_udf = udf(lambda x: eval(x) if x else [], ArrayType(StringType()))

# 토큰화
train_data = train_data.withColumn("filtered_tokens", list_converter_udf("content"))
test_data = test_data.withColumn("filtered_tokens", list_converter_udf("content"))

# CountVectorizer 모델 생성
cv = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features")
cv_model = cv.fit(train_data)

# train data, test data 모두 vectorize
vectorized_train_data = cv_model.transform(train_data)
vectorized_test_data = cv_model.transform(test_data)

# Linear Regression
lr = LogisticRegression(featuresCol="raw_features", labelCol="sentiment")
lr_model = lr.fit(vectorized_test_data)

# test data에 대한 예측 - RegressionEvaluator 사용
predictions = lr_model.transform(vectorized_test_data)

evaluator = RegressionEvaluator(
    labelCol="sentiment", predictionCol="prediction", metricName="rmse"
)

# Root Mean Square Error(RMSE) 계산
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy on test data = {accuracy}")

# Spark 세션 종료
spark.stop()
