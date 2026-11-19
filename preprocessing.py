"""Data preprocessing for the Melbourne Property Price Predictor.

Pipeline stages:
    1. Drop non-predictive columns (index, Address).
    2. Parse the sale Date into Year/Month numeric features.
    3. Impute missing numeric values (median) and categorical values (mode).
    4. Encode categoricals:
         - low cardinality  -> OneHotEncoder
         - high cardinality -> FrequencyEncoder (custom)
    5. Scale all features with StandardScaler.
    6. Reduce dimensionality with PCA (optional, configurable components).

The whole sequence is exposed as an sklearn Pipeline so it can be composed
with the RandomForest model and serialised with joblib.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Price"

# Columns that carry no generalisable signal for a learned model.
DEFAULT_DROP_COLUMNS = ["Unnamed: 0", "Address"]

# Categoricals with a small number of levels -> one-hot encoded.
LOW_CARD_CATS = ["Type", "Method", "Regionname", "CouncilArea"]

# Categoricals with many levels -> frequency encoded to keep dimensionality low.
HIGH_CARD_CATS = ["Suburb", "SellerG"]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set frequency (count share).

    Keeps the feature space compact for high-cardinality columns such as
    Suburb (330) and SellerG (305) where one-hot encoding would explode the
    dimensionality and dilute the PCA signal.
    """

    def __init__(self, unseen_value: float = 0.0):
        self.unseen_value = unseen_value

    def fit(self, X, y=None):
        frame = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.columns_ = list(frame.columns)
        self.freq_ = {}
        for col in self.columns_:
            counts = frame[col].value_counts(normalize=True)
            self.freq_[col] = counts.to_dict()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        out = pd.DataFrame(index=frame.index)
        for col in self.columns_:
            mapping = self.freq_[col]
            out[col] = frame[col].map(mapping).fillna(self.unseen_value).astype(float)
        return out.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


class DateExpander(BaseEstimator, TransformerMixin):
    """Convert the DD/MM/YYYY Date column into Year and Month numeric columns."""

    def __init__(self, date_column: str = "Date"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.date_column in frame.columns:
            parsed = pd.to_datetime(frame[self.date_column], dayfirst=True, errors="coerce")
            frame["SaleYear"] = parsed.dt.year
            frame["SaleMonth"] = parsed.dt.month
            frame.drop(columns=[self.date_column], inplace=True)
        return frame


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop a configurable list of columns from a DataFrame."""

    def __init__(self, columns: Optional[Iterable[str]] = None):
        self.columns = list(columns) if columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        drop = [c for c in self.columns if c in frame.columns]
        return frame.drop(columns=drop)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers at the IQR fences (optional robustness stage)."""

    def __init__(self, columns: Optional[Iterable[str]] = None, factor: float = 1.5):
        self.columns = list(columns) if columns is not None else []
        self.factor = factor

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.bounds_ = {}
        for col in cols:
            if col not in frame.columns:
                continue
            s = frame[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (low, high) in self.bounds_.items():
            if col in frame.columns:
                frame[col] = frame[col].clip(low, high)
        return frame


class DataPreprocessor:
    """Orchestrates the full preprocessing pipeline for the Melbourne dataset."""

    def __init__(
        self,
        drop_columns: Optional[Iterable[str]] = None,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        scale: bool = True,
        use_pca: bool = True,
        pca_components: Optional[int] = None,
        pca_variance: Optional[float] = 0.95,
        cap_outliers: bool = False,
        outlier_columns: Optional[Iterable[str]] = None,
    ):
        self.drop_columns = list(drop_columns) if drop_columns is not None else list(DEFAULT_DROP_COLUMNS)
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.scale = scale
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_variance = pca_variance
        self.cap_outliers = cap_outliers
        self.outlier_columns = list(outlier_columns) if outlier_columns is not None else []
        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_in_: Optional[List[str]] = None

    def _detect_columns(self, X: pd.DataFrame):
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return numeric, categorical

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        frame = X.copy()
        if TARGET in frame.columns:
            frame = frame.drop(columns=[TARGET])

        # Simulate the pre-table stages so column detection matches what the
        # ColumnTransformer will actually receive at fit time.
        probe = ColumnDropper(self.drop_columns).fit_transform(frame)
        probe = DateExpander("Date").fit_transform(probe)
        if self.cap_outliers:
            probe = OutlierCapper(self.outlier_columns).fit_transform(probe)

        numeric, categorical = self._detect_columns(probe)
        low_card = [c for c in LOW_CARD_CATS if c in categorical]
        high_card = [c for c in HIGH_CARD_CATS if c in categorical]
        remaining_cat = [c for c in categorical if c not in low_card and c not in high_card]
        if remaining_cat:
            high_card = high_card + remaining_cat

        numeric_imputer = SimpleImputer(strategy=self.numeric_impute_strategy)
        cat_imputer = SimpleImputer(strategy=self.categorical_impute_strategy)

        encoders = []
        if low_card:
            encoders.append(
                ("onehot", Pipeline([
                    ("impute", cat_imputer),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), low_card)
            )
        if high_card:
            encoders.append(
                ("freq", Pipeline([
                    ("impute", cat_imputer),
                    ("encode", FrequencyEncoder()),
                ]), high_card)
            )

        transformers = [
            ("num", numeric_imputer, numeric),
        ] + encoders

        pre_ct = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=-1)

        steps = [
            ("drop", ColumnDropper(self.drop_columns)),
            ("dates", DateExpander("Date")),
        ]
        if self.cap_outliers:
            steps.append(("outliers", OutlierCapper(self.outlier_columns)))
        steps.append(("features", pre_ct))
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        if self.use_pca:
            if self.pca_components is not None:
                pca = PCA(n_components=self.pca_components, random_state=42)
            else:
                pca = PCA(n_components=self.pca_variance, random_state=42)
            steps.append(("pca", pca))

        self.pipeline_ = Pipeline(steps)
        self.feature_names_in_ = list(frame.columns)
        return self.pipeline_

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pipe = self.pipeline_ if self.pipeline_ is not None else self.build_pipeline(X)
        return pipe.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.pipeline_.transform(X)

    @property
    def pca_(self):
        if self.pipeline_ is None or not self.use_pca:
            return None
        return self.pipeline_.named_steps.get("pca")

    @property
    def explained_variance_ratio_(self):
        pca = self.pca_
        return pca.explained_variance_ratio_ if pca is not None else None

    def summary(self) -> dict:
        return {
            "drop_columns": self.drop_columns,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "scale": self.scale,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "pca_variance": self.pca_variance,
            "cap_outliers": self.cap_outliers,
        }


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET):
    y = df[target].copy()
    X = df.drop(columns=[target])
    return X, y