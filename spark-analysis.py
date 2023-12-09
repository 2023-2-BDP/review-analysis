
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat, lit, udf
from pyspark.sql.types import IntegerType

# Spark 세션 생성
spark = SparkSession.builder.appName("bdp_final").getOrCreate()

# CSV 파일 읽기
reviews = spark.read.csv("hdfs:///user/maria_dev/coupang3.csv", header="true", sep="\t", inferSchema="true")

# 필터링 및 형식 변환
reviews = (
    reviews
    .filter(col('rating').isin(['1', '2', '3', '4', '5']))
    .withColumn("rating", col("rating").cast(IntegerType()))
    .filter((col("headline").isNull() | col("review_content").isNull()) == False)
    .withColumn("headline", when(col("headline").contains('등록된 헤드라인이'), '').otherwise(col("headline")))
    .withColumn("review_content", when(col("review_content").contains('등록된 리뷰내용이'), '').otherwise(col("review_content")))
    .withColumn("content", concat(col("headline"), lit(" "), col("review_content")))
    .dropDuplicates(['rating', 'content'])
)

# 캐싱 적용
reviews.cache()

# Define the split_rating function
def split_rating(rating):
    if rating == 5:
        return 1
    if rating in [1, 2, 3]:
        return 0
    return -1

# Create a UDF
split_rating_udf = udf(split_rating, IntegerType())

# Apply the UDF to the DataFrame
reviews = reviews.withColumn('sentiment', split_rating_udf(reviews['rating']))

# 샘플링 및 합치기
review_count = reviews.count()
reviews_sample_positive = reviews.sampleBy("sentiment", fractions={1: 15000/review_count}, seed=1353)
reviews_sample_negative = reviews.sampleBy("sentiment", fractions={0: 15000/review_count}, seed=1353)
reviews_sample = reviews_sample_positive.union(reviews_sample_negative)

# 샘플링한 데이터로 새로운 DataFrame 생성
(train_df, valid_df, test_df) = reviews.randomSplit([0.6, 0.2, 0.2], seed=1353)

# CSV로 저장
train_df.write.csv("hdfs:///user/maria_dev/coupang3_train", sep="\t", header=True, mode="overwrite")
valid_df.write.csv("hdfs:///user/maria_dev/coupang3_valid", sep="\t", header=True, mode="overwrite")
test_df.write.csv("hdfs:///user/maria_dev/coupang3_test", sep="\t", header=True, mode="overwrite")

