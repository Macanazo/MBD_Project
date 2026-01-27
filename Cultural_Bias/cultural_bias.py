from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.appName("Steam_Cultural_Bias_Analysis").getOrCreate()


df = spark.read.csv(
    "/user/s2578018/all_reviews.csv.gz",
    header=True,
    inferSchema=True,
    quote='"',
    escape='"'
)

df = df.withColumn("is_positive", F.col("voted_up").cast("double"))

df = df.filter(F.col("appid").isNotNull() & F.col("language").isNotNull())

game_stats = df.groupBy("appid").agg(
    F.mean("is_positive").alias("game_global_avg"),
    F.count("*").alias("game_review_count")
)

#When looking at the data, we found a Long Tail distribution. With the number 500 we are selecting the top8% of games, so we can get the global average sentiment that aligns with the central limit theorem.
# Without being skewed by outliers. Meaning thi analysis contains the top 8% of games which represent culturally relevant titles
valid_games = game_stats.filter(F.col("game_review_count") > 500)

df_joined = df.join(valid_games, on="appid", how="inner")

df_bias = df_joined.withColumn("user_bias", F.col("is_positive") - F.col("game_global_avg"))

language_stats = df_bias.groupBy("language").agg(
    F.count("*").alias("total_reviews"),
    F.mean("is_positive").alias("raw_positivity"),
    F.mean("user_bias").alias("avg_cultural_bias"),
    F.stddev("user_bias").alias("bias_stddev")
)

# The median reviews per language is 493000. So we excluded languages with less than 10k reviews to eliminate noise and ensure each cultural group has a significant sample size.
final_stats = language_stats.filter(F.col("total_reviews") > 10000) \
                            .orderBy(F.desc("avg_cultural_bias"))

final_stats.orderBy(F.asc("avg_cultural_bias"))

final_stats.write.csv("USERBIAS", header=True, mode="overwrite")
