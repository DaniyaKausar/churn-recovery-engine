# src/models/experiment_tracker.py
# ==================================
# MLflow tracks every experiment we run.
# This is what real ML teams use in production.
# It answers: "Which model version gave best results?"

import mlflow
import mlflow.xgboost
import os


class ExperimentTracker:
    """
    Tracks model experiments with MLflow.
    
    Every time you train a model, MLflow saves:
    - Parameters used (learning rate, max_depth etc)
    - Metrics achieved (AUC, F1, Recall)
    - The model artifact itself
    - Plots and visualizations
    
    This means you can always go back and compare
    different runs to find the best model.
    """

    def __init__(
        self,
        experiment_name: str = "churn_prediction"
    ):
        # Set where MLflow saves data
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"✅ MLflow tracking: {experiment_name}")

    def log_training_run(
        self,
        params: dict,
        metrics: dict,
        model,
        feature_names: list,
        threshold: float
    ):
        """
        Logs one complete training run to MLflow.
        Call this after every model training.
        """

        with mlflow.start_run(
            run_name="xgboost_optuna_tuned"
        ) as run:

            # Log all hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log custom parameters
            mlflow.log_param(
                "feature_count",
                len(feature_names)
            )
            mlflow.log_param(
                "threshold",
                round(threshold, 4)
            )
            mlflow.log_param(
                "optimization_method",
                "Optuna_30_trials"
            )

            # Log all performance metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(
                        metric_name,
                        metric_value
                    )

            # Log feature names as artifact
            with open("models/feature_names_log.txt", "w") as f:
                for feat in feature_names:
                    f.write(feat + "\n")
            mlflow.log_artifact(
                "models/feature_names_log.txt"
            )

            # Log SHAP plots if they exist
            shap_plots = [
                "models/shap_summary.png",
                "models/shap_importance.png",
                "models/shap_waterfall.png"
            ]
            for plot_path in shap_plots:
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path)

            # Save run ID for reference
            run_id = run.info.run_id

            print(f"\n📊 MLflow Run Logged!")
            print(f"   Run ID: {run_id}")
            print(f"   Experiment: {self.experiment_name}")
            print(f"   AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
            print(f"   F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(
                f"\n   View UI: mlflow ui "
                f"(open http://localhost:5000)"
            )

            return run_id