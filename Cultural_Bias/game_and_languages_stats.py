from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
    .appName("Steam_Stats_Calc") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()


df = spark.read.csv(
    "/user/s2578018/all_reviews.csv.gz",
    header=True,
    inferSchema=True,
    quote='"',
    escape='"'
)


df_clean = df.filter(F.col("appid").cast("int").isNotNull()) \
             .filter(F.col("language").isNotNull())

df_clean = df_clean.filter(
    F.col("language").rlike("^[a-zA-Z_]+$")
)

game_counts = df_clean.groupBy("appid").count().withColumnRenamed("count", "review_count")
lang_counts = df_clean.groupBy("language").count().withColumnRenamed("count", "review_count")

pdf_games = game_counts.select("review_count")

pdf_games.write.csv("plot_data_games", header=True, mode="overwrite")
lang_counts.write.csv("plot_data_langs", header=True, mode="overwrite")
