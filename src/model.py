"""LightGBM Classifier Engine for Pairwise Entity Resolution.

Implements EntityMatcherLGBM wrapping LightGBM binary classification with
ROC-AUC early stopping, feature importance analysis, and model persistence.
"""

from typing import Any, Dict, List, Optional
import os
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from src.features import FEATURE_NAMES


class EntityMatcherLGBM:
    """LightGBM Binary Classifier for Candidate Match Scoring."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        num_leaves: int = 63,
        max_depth: int = 7,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.classifier: Optional[lgb.LGBMClassifier] = None
        self.booster_: Optional[lgb.Booster] = None
        self.best_iteration_: int = -1
        self.val_auc_: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "EntityMatcherLGBM":
        """Fit binary classifier with early stopping on validation ROC-AUC."""
        self.classifier = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

        callbacks: List[Any] = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose),
        ]
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))
        else:
            callbacks.append(lgb.log_evaluation(period=0))

        feature_names_list = list(FEATURE_NAMES)
        self.classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["val"],
            eval_metric="auc",
            callbacks=callbacks,
            feature_name=feature_names_list,
        )

        self.booster_ = self.classifier.booster_
        self.best_iteration_ = self.classifier.best_iteration_ or self.n_estimators

        val_probs = self.predict_proba(X_val)
        if len(np.unique(y_val)) > 1:
            self.val_auc_ = float(roc_auc_score(y_val, val_probs))
        else:
            self.val_auc_ = 1.0

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match P(match) in [0.0, 1.0]."""
        if self.booster_ is not None:
            # Booster predict gives raw binary probability for class 1 directly
            preds = self.booster_.predict(X, num_iteration=self.best_iteration_)
            return np.asarray(preds, dtype=np.float32)
        elif self.classifier is not None:
            return self.classifier.predict_proba(X)[:, 1].astype(np.float32)
        else:
            raise RuntimeError("Model is not fitted yet.")

    def get_feature_importances(self, importance_type: str = "gain") -> Dict[str, float]:
        """Return mapping of feature names to importance scores."""
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted yet.")

        importances = self.booster_.feature_importance(importance_type=importance_type)
        return {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

    def save_model(self, filepath: str) -> None:
        """Save booster model to text file."""
        if self.booster_ is None:
            raise RuntimeError("No trained model to save.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.booster_.save_model(filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "EntityMatcherLGBM":
        """Load trained booster model from text file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        instance = cls()
        instance.booster_ = lgb.Booster(model_file=filepath)
        instance.best_iteration_ = instance.booster_.best_iteration
        return instance
