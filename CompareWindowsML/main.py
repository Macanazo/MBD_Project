from pyspark.sql import SparkSession
from data_processor import SteamDataProcessor
from model_factory import ModelFactory
from results_manager import ResultsManager
from pyspark.ml.evaluation import RegressionEvaluator
import config

def main():
    spark = SparkSession.builder.appName("SteamAnalysis").getOrCreate()
    
    # Load and clean data
    processor = SteamDataProcessor(spark, config.DATA_PATH, config.MIN_REVIEWS_THRESHOLD)
    raw_data = processor.load_clean_data()
    
    # Engineer features and predictors
    ml_ready_data = processor.get_features_and_label(raw_data)
    train, test = ml_ready_data.randomSplit(config.TRAIN_TEST_SPLIT, seed=42)
    
    # Train the models
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    factory = ModelFactory(train, evaluator)
    
    best_results = {}
    # Use simple fit to compare models efficiently
    models_to_compare = [
        ("LinearRegression", factory.build_lr()[0]),
        ("RandomForest", factory.build_rf()[0]),
        ("XGBoost", factory.build_xgboost()[0])
    ]

    # Evaluate and collect metrics
    summary_metrics = []
    for name, model_obj in models_to_compare:
        print(f"Training {name}...")
        model = model_obj.fit(train)
        best_results[name] = model
        
        preds = model.transform(test)
        
        # Create unique evaluators for each call to fix the duplicate metric bug
        r2 = RegressionEvaluator(metricName="r2", labelCol="label").evaluate(preds)
        rmse = RegressionEvaluator(metricName="rmse", labelCol="label").evaluate(preds)
        
        summary_metrics.append((name, r2, rmse))
    
    # Extract Feature Importances/Coefficients
    feature_names = ["m1_ratio", "log_m1_count"] 
    importance_list = []
    for name, model in best_results.items():
        try:
            if name == "LinearRegression":
                weights = model.coefficients.toArray()
            elif name == "RandomForest":
                weights = model.featureImportances.toArray()
            else: # XGBoost
                
                booster = model.get_booster()
               
                importance_dict = booster.get_score(importance_type="weight")
                weights = [importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_names))]
            for i, w in enumerate(weights):
                if i < len(feature_names):
                    importance_list.append((name, feature_names[i], float(w)))
        except Exception as e:
            print(f"Could not get importance for {name}: {e}")

    # Save
    manager = ResultsManager(spark)
    manager.save_metrics_to_csv(summary_metrics, config.METRICS_CSV_PATH)
    manager.save_importance_to_csv(importance_list, config.IMPORTANCE_PATH)
    print(manager.save_summary_report(summary_metrics, ml_ready_data.count(), config.REPORT_PATH))

if __name__ == "__main__":
    main()
