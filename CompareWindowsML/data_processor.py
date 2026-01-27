from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler

# Class to handle loading, cleaning, and feature engineering of the data
class SteamDataProcessor:
    def __init__(self, spark, data_path, min_reviews):
        self.spark = spark
        self.data_path = data_path
        self.min_reviews = min_reviews

    def load_clean_data(self):
        # Filter for only non early access reviews and select relevant columns
        print("[PROCESS] Loading raw data from HDFS...")
        df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
        return df.filter(F.col("written_during_early_access") == 0).select(
            "appid", "voted_up", 
            F.from_unixtime("timestamp_created").cast("timestamp").alias("date")
        ).dropna()

    def get_features_and_label(self, df):
        # Identifie release dates and calculate review windows
        print("[PROCESS] Engineering temporal features...")
        window = Window.partitionBy("appid").orderBy("date")
        df_ranked = df.withColumn("rank", F.rank().over(window))
        
        releases = df_ranked.filter(F.col("rank") == 1).select(
            F.col("appid").alias("ref_id"), F.col("date").alias("release_date")
        )
        
        timed_df = df.join(releases, df.appid == releases.ref_id) \
                     .withColumn("days_since", F.datediff("date", "release_date"))
        
        # Aggregate 1st Month predictors
        m1 = timed_df.filter((F.col("days_since") >= 0) & (F.col("days_since") <= 30)) \
                     .groupBy("appid").agg(
                         F.avg("voted_up").alias("m1_ratio"),
                         F.count("voted_up").alias("m1_count")
                     ).filter(F.col("m1_count") >= self.min_reviews) \
                     .withColumn("log_m1_count", F.log1p("m1_count"))
        
        # Aggregate 2nd to 12th Month predictors
        y1 = timed_df.filter((F.col("days_since") > 30) & (F.col("days_since") <= 365)) \
                     .groupBy("appid").agg(F.avg("voted_up").alias("label"))
        
        final_set = m1.join(y1, "appid").dropna()
        
        # Vectorize features for Spark
        assembler = VectorAssembler(inputCols=["m1_ratio", "log_m1_count"], outputCol="features")
        return assembler.transform(final_set).select("label", "features", "m1_count")