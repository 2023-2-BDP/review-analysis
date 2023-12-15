# 필요한 라이브러리 import
# fmt:off
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat, lit, udf, substring
from pyspark.sql.types import IntegerType


# Spark 세션 생성
spark = SparkSession.builder.appName("bdp_final").getOrCreate()

# CSV 파일 읽기(coupang1부터 3까지 차례대로 수행)
reviews = spark.read.csv("hdfs:///user/maria_dev/coupang1.csv", header="true", sep="\t", inferSchema="true")
# reviews = spark.read.csv("hdfs:///user/maria_dev/coupang2.csv", header="true", sep="\t", inferSchema="true")
# reviews = spark.read.csv("hdfs:///user/maria_dev/coupang3.csv", header="true", sep="\t", inferSchema="true")

# 필터링 및 형식 변환
reviews = (
    reviews

    # 앞의 30글자만 남기기
    .withColumn('review_content', substring(reviews['review_content'], 1, 30))

    # 평점이 1~5 사이인 데이터만 남기기 (데이터 정제)
    .filter(col('rating').isin(['1', '2', '3', '4', '5']))
    .withColumn("rating", col("rating").cast(IntegerType()))

    # headline, review_content가 null인 데이터 제거
    .filter((col("headline").isNull() | col("review_content").isNull()) == False)
    .withColumn("headline", when(col("headline").contains('등록된 헤드라인이'),'').otherwise(col("headline")))
    .withColumn("review_content", when(col("review_content").contains('등록된 리뷰내용이'), '').otherwise(col("review_content")))

    # headline, review_content를 하나의 컬럼으로 합치기
    .withColumn("content", concat(col("headline"), lit(" "), col("review_content")))

    # 중복 데이터 제거
    .dropDuplicates(['rating', 'content'])
)

# 캐싱 적용
reviews.cache()

# 평점 분류(긍정, 부정)
def split_rating(rating):
    if rating == 5:
        return 1
    if rating in [1, 2, 3]:
        return 0
    return -1

# UDF(함수 실행)
split_rating_udf = udf(split_rating, IntegerType())

# Apply the UDF to the DataFrame
reviews = reviews.withColumn('sentiment', split_rating_udf(reviews['rating']))

# 부정적 리뷰 개수 count
negative_count = reviews.filter(reviews['sentiment'] == 0).count()
negative_count = float(negative_count)

# 샘플링 후 합치기
# 밸런스 맞추기 위해 부정적 리뷰 개수만큼 긍정적 리뷰 샘플링
review_count = reviews.count()
reviews_sample_positive = reviews.sampleBy("sentiment", fractions={1: negative_count/review_count}, seed=1353)
reviews_sample_negative = reviews.sampleBy("sentiment", fractions={0: negative_count/review_count}, seed=1353)
reviews_sample = reviews_sample_positive.union(reviews_sample_negative)

# 샘플데이터로 데이터프레임 생성
(train_df, valid_df, test_df) = reviews.randomSplit([0.6, 0.2, 0.2], seed=1353)

# Write to CSV(mode = "append"를 통해 coupang1.csv~coupang3.csv까지 데이터 뒤에 이어 붙이기)
train_df.write.csv("hdfs:///user/maria_dev/coupang_train", sep="\t", header=True, mode="append")
valid_df.write.csv("hdfs:///user/maria_dev/coupang_valid", sep="\t", header=True, mode="append")
test_df.write.csv("hdfs:///user/maria_dev/coupang_test", sep="\t", header=True, mode="append")
