from pyspark.sql import functions as F

def run_playtime_purchase_effect(spark, input_path, out_base):
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    df = df.select(
        "appid",
        "author_playtime_at_review",
        "steam_purchase",
        "voted_up"
    ).dropna()

    # playtime as double
    df = df.withColumn(
        "playtime",
        F.col("author_playtime_at_review").cast("double")
    )

    # normalize voted_up to 0/1
    df = df.withColumn(
        "voted_up_i",
        F.when(F.lower(F.col("voted_up")).isin("1", "true", "t", "yes"), 1)
         .when(F.lower(F.col("voted_up")).isin("0", "false", "f", "no"), 0)
         .otherwise(F.col("voted_up").cast("int"))
    )

    # normalize steam_purchase to 0/1
    df = df.withColumn(
        "steam_purchase_i",
        F.when(F.lower(F.col("steam_purchase")).isin("1", "true", "t", "yes"), 1)
         .when(F.lower(F.col("steam_purchase")).isin("0", "false", "f", "no"), 0)
         .otherwise(F.col("steam_purchase").cast("int"))
    )

    df = df.filter(
        F.col("playtime").isNotNull() &
        F.col("voted_up_i").isin(0, 1) &
        F.col("steam_purchase_i").isin(0, 1)
    )

    df = df.withColumn(
        "playtime_bucket",
        F.when(F.col("playtime") < 2, "<2h")
         .when(F.col("playtime") < 10, "2-10h")
         .when(F.col("playtime") < 50, "10-50h")
         .otherwise("50h+")
    )

    result = (
        df.groupBy("playtime_bucket")
          .agg(
              F.avg("steam_purchase_i").alias("purchase_rate"),
              F.avg("voted_up_i").alias("positive_rate"),
              F.count("*").alias("review_count")
          )
          .orderBy("playtime_bucket")
    )

    output_path = f"{out_base}/playtime_purchase_effect"
    result.write.mode("overwrite").csv(output_path, header=True)

    return output_path
