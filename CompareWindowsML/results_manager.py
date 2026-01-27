from pyspark.sql import functions as F

class ResultsManager:
    def __init__(self, spark):
        self.spark = spark

    def save_metrics_to_csv(self, metrics_list, path):
        # Create DataFrame from results
        df = self.spark.createDataFrame(metrics_list, ["Model", "R2", "RMSE"])
        
        # Save to hdfs
        df.coalesce(1).write.csv(
            path, 
            header=True, 
            mode="overwrite"
        )
        print(f"[STORAGE] Metrics CSV folder saved to HDFS: {path}")
    # Save feature importance to CSV
    def save_importance_to_csv(self, importance_list, path):
        # Create DataFrame from importance rankings
        df = self.spark.createDataFrame(importance_list, ["Model", "Feature", "Weight"])
        
        # Save to hdfs
        df.coalesce(1).write.csv(
            path, 
            header=True, 
            mode="overwrite"
        )
        print(f"[STORAGE] Feature Importance CSV folder saved to HDFS: {path}")
    #Save summary report as text file 
    def save_summary_report(self, metrics_list, total_games, path):
        report_lines = [
            "Final Model Comparison Report",
            f"Games in Study: {total_games}",
            "Model\t\tR2\t\tRMSE" 
        ]
        for name, r2, rmse in metrics_list:
            report_lines.append(f"{name}\t{r2}\t{rmse}")
        
        report_str = "\n".join(report_lines)
        report_df = self.spark.createDataFrame([(report_str,)], ["report"])
        
        # Save to hdfs
        report_df.coalesce(1).write.mode("overwrite").text(path)
        return report_str