from pyspark.sql import SparkSession
from playtime_analysis import run_playtime_purchase_effect

def main():
    spark = SparkSession.builder \
        .appName("PlaytimePurchaseEffect") \
        .getOrCreate()

    input_path = "hdfs:///user/s2578018/all_reviews.csv.gz"
    out_base = "hdfs:///user/s2523477/project_outputs"

    out_path = run_playtime_purchase_effect(spark, input_path, out_base)
    print("Results written to:", out_path)

    spark.stop()

if __name__ == "__main__":
    main()
