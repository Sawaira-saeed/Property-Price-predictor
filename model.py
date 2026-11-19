"""Model training and evaluation for the Property Price Predictor.

Combines the DataPreprocessor with a RandomForestRegressor inside a single
sklearn Pipeline, evaluates with R2 / RMSE / MAE, exposes feature importances
mapped back to PCA components, and supports joblib persistence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import TARGET, DataPreprocessor, split_features_target


@dataclass
class Metrics:
    r2_train: float
    r2_test: float
    rmse_train: float
    rmse_test: float
    mae_train: float
    mae_test: float

    def as_dict(self) -> dict:
        return {
            "R2 (train)": self.r2_train,
            "R2 (test)": self.r2_test,
            "RMSE (train)": self.rmse_train,
            "RMSE (test)": self.rmse_test,
            "MAE (train)": self.mae_train,
            "MAE (test)": self.mae_test,
        }


@dataclass
class TrainResult:
    metrics: Metrics
    n_features_in: int
    n_components: Optional[int]
    explained_variance: Optional[np.ndarray]
    importances: Optional[np.ndarray]
    component_labels: Optional[list] = None


class ModelTrainer:
    """End-to-end trainer: preprocessing -> PCA -> RandomForest."""

    def __init__(
        self,
        preprocessor: Optional[DataPreprocessor] = None,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.preprocessor = preprocessor or DataPreprocessor()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.pipeline_: Optional[Pipeline] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.last_result: Optional[TrainResult] = None

    def _build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def prepare_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = split_features_target(df, TARGET)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self, df: Optional[pd.DataFrame] = None, test_size: float = 0.2) -> TrainResult:
        if df is not None:
            self.prepare_split(df, test_size=test_size)

        if self.X_train is None:
            raise RuntimeError("No training data. Call prepare_split or pass df.")

        pre_pipe = self.preprocessor.build_pipeline(self.X_train)
        self.pipeline_ = Pipeline([
            ("preprocess", pre_pipe),
            ("regressor", self._build_estimator()),
        ])

        self.pipeline_.fit(self.X_train, self.y_train)

        pred_train = self.pipeline_.predict(self.X_train)
        pred_test = self.pipeline_.predict(self.X_test)

        metrics = Metrics(
            r2_train=r2_score(self.y_train, pred_train),
            r2_test=r2_score(self.y_test, pred_test),
            rmse_train=np.sqrt(mean_squared_error(self.y_train, pred_train)),
            rmse_test=np.sqrt(mean_squared_error(self.y_test, pred_test)),
            mae_train=mean_absolute_error(self.y_train, pred_train),
            mae_test=mean_absolute_error(self.y_test, pred_test),
        )

        regressor = self.pipeline_.named_steps["regressor"]
        importances = regressor.feature_importances_

        pca = self.preprocessor.pca_
        if pca is not None:
            explained = pca.explained_variance_ratio_
            n_comp = pca.n_components_
            labels = [f"PC{i + 1}" for i in range(n_comp)]
        else:
            explained = None
            n_comp = importances.shape[0]
            labels = [f"F{i + 1}" for i in range(n_comp)]

        self.last_result = TrainResult(
            metrics=metrics,
            n_features_in=self.X_train.shape[1],
            n_components=n_comp,
            explained_variance=explained,
            importances=importances,
            component_labels=labels,
        )
        return self.last_result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Model is not trained yet.")
        return self.pipeline_.predict(X)

    def predict_one(self, record: dict) -> float:
        frame = pd.DataFrame([record])
        return float(self.predict(frame)[0])

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Nothing to save: model not trained.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline_,
                "preprocessor": self.preprocessor,
                "feature_names": self.preprocessor.feature_names_in_,
                "result": self.last_result,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ModelTrainer":
        bundle = joblib.load(path)
        trainer = cls(preprocessor=bundle["preprocessor"])
        trainer.pipeline_ = bundle["pipeline"]
        trainer.last_result = bundle.get("result")
        return trainer