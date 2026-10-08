"""
models.py -- Phase 4: baselines, M1 (NB GLM + logistic), M2 (M1 + temperature
spline), M3 (LightGBM). All scalers / spline knots are fitted on the training
rows passed to .fit() only.

Every model exposes:
  fit(df_train)            df has feature columns + y_reg + y_cls
  predict(df) -> DataFrame mean, p10, p90, prob
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.stats import nbinom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
ALPHA_MIN = 1e-3          # NB dispersion below this -> Poisson


class Design:
    """Standardised linear columns (+ optional B-spline on one raw column, df=3).
    Scaler and spline knots are learnt in fit()."""

    def __init__(self, feats, spline_col=None):
        self.feats, self.spline_col = list(feats), spline_col

    def fit(self, df):
        self.scaler = StandardScaler().fit(df[self.feats])
        self.spline_info = None
        if self.spline_col:
            self.spline_info = dmatrix(f"bs({self.spline_col}, df=3) - 1", df,
                                       return_type="dataframe").design_info
        return self

    def transform(self, df):
        X = pd.DataFrame(self.scaler.transform(df[self.feats]), columns=self.feats, index=df.index)
        if self.spline_info is not None:
            # values outside the training range are clipped to it (bs extrapolates badly)
            lo, hi = self._range
            sub = df[[self.spline_col]].clip(lo, hi)
            S = build_design_matrices([self.spline_info], sub, return_type="dataframe")[0]
            S.columns = [f"spl{i}" for i in range(S.shape[1])]
            S.index = df.index
            X = pd.concat([X, S], axis=1)
        return X

    def fit_transform(self, df):
        if self.spline_col:
            self._range = (df[self.spline_col].min(), df[self.spline_col].max())
        return self.fit(df).transform(df)


class GLMModel:
    """M1 / M2: negative-binomial GLM regressor + logistic-regression classifier."""

    def __init__(self, feats, spline_col=None, name="M1"):
        self.feats, self.spline_col, self.name = list(feats), spline_col, name

    def fit(self, df):
        self.design = Design(self.feats, self.spline_col)
        X = sm.add_constant(self.design.fit_transform(df), has_constant="add")
        y = df.y_reg.values
        self.family_used, self.alpha = "negbin", np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                nb = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=500)
                self.alpha = float(nb.params.iloc[-1])
            except Exception:                           # noqa: BLE001
                self.alpha = np.nan
            if not np.isfinite(self.alpha) or self.alpha < ALPHA_MIN:
                self.family_used = "poisson"
                self.reg = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            else:
                self.reg = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        yc = df.y_cls.values
        Xc = self.design.transform(df)
        if len(np.unique(yc)) < 2:
            self.cls, self.base_rate = None, float(yc.mean())
        else:
            self.cls = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xc, yc)
        return self

    def predict(self, df):
        X = sm.add_constant(self.design.transform(df), has_constant="add")
        mu = np.clip(self.reg.predict(X), 0, 1e6)
        if self.family_used == "negbin":
            n = 1.0 / self.alpha
            p = n / (n + mu)
            p10, p90 = nbinom.ppf(0.1, n, p), nbinom.ppf(0.9, n, p)
        else:
            from scipy.stats import poisson
            p10, p90 = poisson.ppf(0.1, mu), poisson.ppf(0.9, mu)
        prob = (self.cls.predict_proba(self.design.transform(df))[:, 1]
                if self.cls is not None else np.full(len(df), self.base_rate))
        return pd.DataFrame({"mean": mu, "p10": p10, "p90": p90, "prob": prob}, index=df.index)


class LGBMModel:
    """M3 challenger: shallow LightGBM (Poisson regressor, quantile range, binary classifier)."""

    PARAMS = dict(max_depth=3, num_leaves=7, min_child_samples=20, learning_rate=0.05,
                  n_estimators=300, random_state=SEED, verbose=-1, deterministic=True,
                  force_col_wise=True)

    def __init__(self, feats, name="M3"):
        self.feats, self.name = list(feats), name

    def fit(self, df):
        import lightgbm as lgb
        X, y = df[self.feats], df.y_reg
        self.reg = lgb.LGBMRegressor(objective="poisson", **self.PARAMS).fit(X, y)
        self.q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **self.PARAMS).fit(X, y)
        self.q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **self.PARAMS).fit(X, y)
        yc = df.y_cls
        if yc.nunique() < 2 or yc.sum() < 2:
            self.cls, self.base_rate = None, float(yc.mean())
        else:
            self.cls = lgb.LGBMClassifier(objective="binary", **self.PARAMS).fit(X, yc)
        return self

    def predict(self, df):
        X = df[self.feats]
        mu = np.clip(self.reg.predict(X), 0, None)
        lo, hi = np.clip(self.q10.predict(X), 0, None), np.clip(self.q90.predict(X), 0, None)
        p10, p90 = np.minimum(lo, mu), np.maximum(hi, mu)      # keep mean inside its range
        prob = self.cls.predict_proba(X)[:, 1] if self.cls is not None else np.full(len(df), self.base_rate)
        return pd.DataFrame({"mean": mu, "p10": p10, "p90": p90, "prob": prob}, index=df.index)


# ---------------- baselines (no fitting) ----------------
def baseline_predictions(full, rows, horizon=4):
    """full: whole table with targets (sorted, default index); rows: index of test rows.
    seasonal naive: cases at the target week one year earlier (t+4-52) -- known at t.
    persistence:    cases at t.
    endemic channel (classifier): prob = 1 if this week is already above the surge line."""
    cases = full.cases
    seas = cases.shift(52 - horizon)          # row t gets cases[t + 4 - 52]
    out = {
        "seasonal_naive": pd.DataFrame({"mean": seas.loc[rows]}),
        "persistence": pd.DataFrame({"mean": cases.loc[rows]}),
        "endemic_channel": pd.DataFrame({"prob": full.above.loc[rows]}),
    }
    return out


def make_model(key):
    """key -> unfitted model. Simplest first (order used for model selection)."""
    from src import dengue_features as F
    return {
        "M1_A": lambda: GLMModel(F.M1_A, name="M1"),
        "M1_B": lambda: GLMModel(F.M1_B, name="M1"),
        "M2_B": lambda: GLMModel(F.M1_A + ["rh_mean_4"], spline_col="tmean_mean_lag4_12", name="M2"),
        "M3_A": lambda: LGBMModel(F.SET_A, name="M3"),
        "M3_B": lambda: LGBMModel(F.SET_B, name="M3"),
    }[key]()


MODEL_KEYS = ["M1_A", "M1_B", "M2_B", "M3_A", "M3_B"]
COMPLEXITY = {"M1": 1, "M2": 2, "M3": 3}
