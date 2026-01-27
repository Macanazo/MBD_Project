from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import config

# Factory class to build and compare models (LR, RF, XGBoost)
class ModelFactory:
    def __init__(self, train_data, evaluator):
        self.train_data = train_data
        self.evaluator = evaluator

    def build_lr(self):
        # Linear Regression with elastic net penlties
        lr = LinearRegression(
            featuresCol="features", 
            labelCol="label", 
            weightCol="m1_count",
            regParam=config.MODEL_PARAMS["LinearRegression"]["regParam"],
            elasticNetParam=config.MODEL_PARAMS["LinearRegression"]["elasticNetParam"]
        )
        return lr, None 

    def build_rf(self):
        # Random Forest Regressor using config params
        rf = RandomForestRegressor(
            featuresCol="features", 
            labelCol="label",
            numTrees=config.MODEL_PARAMS["RandomForest"]["numTrees"],
            maxDepth=config.MODEL_PARAMS["RandomForest"]["maxDepth"]
        )
        return rf, None 

    def build_xgboost(self):
        # XGBOOST Regressor using config params
        from xgboost.spark import SparkXGBRegressor
        
        xgb = SparkXGBRegressor(
            features_col="features", 
            label_col="label",
            importance_type='gain', # Added to enable importance extraction
            max_depth=config.MODEL_PARAMS["XGBoost"]["max_depth"],
            learning_rate=config.MODEL_PARAMS["XGBoost"]["learning_rate"],
            n_estimators=config.MODEL_PARAMS["XGBoost"]["n_estimators"]
        )
        print("[MODEL] XGBoost model constructed with Simple Fit params.")
        return xgb, None 

    def run_comparison(self, model_tuples):
        # Simple fit for each model in the comparison queue
        best_models = {}
        for name, (model, grid) in model_tuples:
            print(f"[MODEL] Training {name} with Simple Fit")
            
            best_models[name] = model.fit(self.train_data)
            
        return best_models