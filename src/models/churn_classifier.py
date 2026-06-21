# src/models/churn_classifier.py

import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
import numpy as np
import pandas as pd
import joblib
import os
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class ChurnClassifier:

    def __init__(self):
        self.model = None
        self.best_params = None
        self.threshold = 0.5
        self.feature_names = None

    def _objective(self, trial, X_train, y_train):
        """XGBoost 2.0+ compatible objective"""
        
        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators', 100, 400
            ),
            'max_depth': trial.suggest_int(
                'max_depth', 3, 7
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.2, log=True
            ),
            'subsample': trial.suggest_float(
                'subsample', 0.6, 1.0
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.6, 1.0
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight', 1, 10
            ),
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight', 1.0, 4.0
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha', 0, 1.0
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1.0, 5.0
            ),
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0,
            'device': 'cpu'
        }

        # XGBoost 2.0 compatible way to train + evaluate
        # We do manual cross validation instead of sklearn's
        X_arr = X_train.values if hasattr(
            X_train, 'values'
        ) else X_train
        y_arr = y_train.values if hasattr(
            y_train, 'values'
        ) else y_train

        skf = StratifiedKFold(
            n_splits=5, 
            shuffle=True, 
            random_state=42
        )
        
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_fold_train = X_arr[train_idx]
            X_fold_val = X_arr[val_idx]
            y_fold_train = y_arr[train_idx]
            y_fold_val = y_arr[val_idx]

            # Use XGBoost DMatrix for direct training
            # This avoids sklearn wrapper issues in XGBoost 2.0
            dtrain = xgb.DMatrix(
                X_fold_train, 
                label=y_fold_train
            )
            dval = xgb.DMatrix(
                X_fold_val, 
                label=y_fold_val
            )

            # Convert params for xgb.train format
            booster_params = {
                k: v for k, v in params.items() 
                if k not in ['n_estimators', 'random_state']
            }
            booster_params['objective'] = 'binary:logistic'
            booster_params['eval_metric'] = 'auc'
            booster_params['seed'] = 42

            booster = xgb.train(
                booster_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'val')],
                verbose_eval=False
            )

            val_preds = booster.predict(dval)
            auc = roc_auc_score(y_fold_val, val_preds)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    def tune_and_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 30
    ):
        print("🔍 Finding best XGBoost settings with Optuna...")
        print(f"   Running {n_trials} trials "
              f"(this takes 2-3 minutes)...\n")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            lambda trial: self._objective(
                trial, X_train, y_train
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        print(f"\n✅ Optimization complete!")
        print(f"   Best AUC score: {study.best_value:.4f}")
        print(f"\n   Best parameters found:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")

        self.best_params = study.best_params

        # Train final model using sklearn wrapper
        # but with XGBoost 2.0 compatible params
        final_params = {
            **self.best_params,
            'objective': 'binary:logistic',
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0,
            'device': 'cpu',
            'use_label_encoder': False
        }

        print(f"\n🏋️  Training final model...")

        # Train using DMatrix + xgb.train for compatibility
        X_arr = X_train.values if hasattr(
            X_train, 'values'
        ) else X_train
        y_arr = y_train.values if hasattr(
            y_train, 'values'
        ) else y_train

        dtrain = xgb.DMatrix(X_arr, label=y_arr)

        booster_params = {
            k: v for k, v in final_params.items()
            if k not in [
                'n_estimators', 
                'random_state',
                'use_label_encoder'
            ]
        }
        booster_params['seed'] = 42

        n_estimators = self.best_params.get(
            'n_estimators', 200
        )

        self.booster = xgb.train(
            booster_params,
            dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False
        )

        # Also create sklearn wrapper for SHAP compatibility
        self.model = xgb.XGBClassifier(
            **{
                k: v for k, v in final_params.items()
                if k not in ['use_label_encoder']
            }
        )
        self.model.fit(X_train, y_train)

        self.feature_names = list(X_train.columns)

        print(f"✅ Model trained successfully!")
        return self.model

    def _get_proba(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Get probabilities - handles XGBoost 2.0 compatibility
        Uses DMatrix directly to avoid n_classes_ issue
        """
        X_arr = X.values if hasattr(X, 'values') else X
        dmatrix = xgb.DMatrix(X_arr)
        probs = self.booster.predict(dmatrix)
        return probs

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:

        print("\n🎯 Optimizing prediction threshold...")

        probs = self._get_proba(X_val)

        precisions, recalls, thresholds = (
            precision_recall_curve(y_val, probs)
        )

        f1_scores = (
            2 * precisions * recalls /
            (precisions + recalls + 1e-8)
        )

        best_idx = np.argmax(f1_scores)
        self.threshold = float(thresholds[best_idx])

        # Default threshold performance
        default_preds = (probs >= 0.5).astype(int)
        default_report = classification_report(
            y_val, default_preds, output_dict=True
        )
        print(f"   Default threshold (0.5) performance:")
        print(f"   F1 for churn class: "
              f"{default_report['1']['f1-score']:.4f}")

        # Optimal threshold performance
        optimal_preds = (
            probs >= self.threshold
        ).astype(int)
        optimal_report = classification_report(
            y_val, optimal_preds, output_dict=True
        )
        print(f"\n   Optimal threshold "
              f"({self.threshold:.3f}) performance:")
        print(f"   F1 for churn class: "
              f"{optimal_report['1']['f1-score']:.4f}")

        improvement = (
            optimal_report['1']['f1-score'] -
            default_report['1']['f1-score']
        ) * 100
        print(f"\n   ✅ Improvement: +{improvement:.2f}% F1")
        print(f"   ✅ Optimal threshold: {self.threshold:.3f}")

        return self.threshold

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:

        print("\n📊 Final Model Evaluation on Test Set:")
        print("=" * 45)

        probs = self._get_proba(X_test)
        preds = (probs >= self.threshold).astype(int)

        auc = roc_auc_score(y_test, probs)
        avg_precision = average_precision_score(y_test, probs)
        report = classification_report(
            y_test, preds, output_dict=True
        )
        cm = confusion_matrix(y_test, preds)

        metrics = {
            'auc_roc': round(auc, 4),
            'avg_precision': round(avg_precision, 4),
            'precision': round(
                report['1']['precision'], 4
            ),
            'recall': round(report['1']['recall'], 4),
            'f1_score': round(
                report['1']['f1-score'], 4
            ),
            'accuracy': round(report['accuracy'], 4),
            'threshold': round(self.threshold, 4)
        }

        print(f"\n   AUC-ROC Score:     {metrics['auc_roc']}")
        print(f"   Avg Precision:     {metrics['avg_precision']}")
        print(f"   Precision:         {metrics['precision']}")
        print(f"   Recall:            {metrics['recall']}")
        print(f"   F1 Score:          {metrics['f1_score']}")
        print(f"   Accuracy:          {metrics['accuracy']}")
        print(f"   Threshold used:    {metrics['threshold']}")

        tn, fp, fn, tp = cm.ravel()
        print(f"\n   Confusion Matrix:")
        print(f"   ┌─────────────────────────────────┐")
        print(f"   │              Predicted           │")
        print(f"   │         No Churn  |  Churn       │")
        print(f"   │ No Churn  {tn:5d}  |  {fp:5d}       │")
        print(f"   │ Churn     {fn:5d}  |  {tp:5d}       │")
        print(f"   └─────────────────────────────────┘")

        print(f"\n   In plain English:")
        print(f"   ✅ Correctly identified {tp} churners")
        print(f"   ✅ Correctly identified {tn} loyals")
        print(f"   ⚠️  Missed {fn} churners")
        print(f"   ⚠️  Wrong alarm {fp} times")

        avg_monthly = 64.76
        revenue_saved = tp * avg_monthly * 12
        revenue_missed = fn * avg_monthly * 12

        print(f"\n   💰 Business Impact:")
        print(f"   Revenue potentially saved: "
              f"${revenue_saved:,.0f}/year")
        print(f"   Revenue still at risk: "
              f"${revenue_missed:,.0f}/year")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:

        # Get importance from booster directly
        scores = self.booster.get_fscore()

        # ✅ FIX: Map f0, f1, f2... back to real feature names
        feature_map = {
            f"f{i}": name 
            for i, name in enumerate(self.feature_names)
        }
        
        real_names = [
            feature_map.get(f, f) for f in scores.keys()
        ]

        importance_df = pd.DataFrame({
            'feature': real_names,
            'importance': list(scores.values())
        })

        # Normalize
        total = importance_df['importance'].sum()
        importance_df['importance'] = (
            importance_df['importance'] / total
        )

        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        print("\n🏆 Top 10 Most Important Features:")
        print("=" * 40)
        for i, row in importance_df.head(10).iterrows():
            bar = "█" * int(row['importance'] * 200)
            print(f"   {i+1:2d}. {row['feature']:<25} "
                f"{row['importance']:.4f} {bar}")

        return importance_df

    def save_model(self):
        os.makedirs('models', exist_ok=True)

        # Save booster (most compatible format)
        self.booster.save_model('models/churn_model.json')

        # Save sklearn wrapper too
        joblib.dump(self.model, 'models/churn_model_sklearn.pkl')
        joblib.dump(self.threshold, 'models/threshold.pkl')
        joblib.dump(self.best_params, 'models/best_params.pkl')
        joblib.dump(self.booster, 'models/booster.pkl')

        print(f"\n💾 Booster saved to models/churn_model.json")
        print(f"💾 Threshold saved to models/threshold.pkl")