"""
dengue_features.py -- Phase 3 (named to avoid clashing with MonsoonWatch's src/features/ package): features at week t, using only rows <= t.

Every feature is a trailing window (rolling / shift >= 0) over the weekly table
sorted by time, so row t never sees row t+1. tests/test_leakage.py enforces this.
"""
import numpy as np
import pandas as pd

WEATHER = ["tmin_mean_4", "tmin_mean_8", "tmean_mean_lag4_12", "heat_above_18_8",
           "rain_sum_4", "rain_sum_8", "rh_mean_4", "warm_wet_weeks_8"]
CASE = ["log_cases_0", "log_cases_1", "log_cases_2", "log_cases_3", "growth_2"]
SET_A = CASE + ["log_last_season", "woy_sin", "woy_cos"]
SET_B = SET_A + WEATHER

# M1 is capped at 8 features. Set A: drop log_cases_2/3 (near-duplicates of 0/1).
# Set B: add the two weather signals the Nepal literature points to, chosen a priori
# (not by looking at this data): minimum temperature ~2 months back (Chitwan NB
# study) -> tmin_mean_8, and humidity -> rh_mean_4.
M1_A = ["log_cases_0", "log_cases_1", "growth_2", "log_last_season", "woy_sin", "woy_cos"]
M1_B = M1_A + ["tmin_mean_8", "rh_mean_4"]

WARM_C, WET_RH = 25.0, 70.0      # defaults; warm cut-off is re-tuned per fold (see warm_cutoff)


def warm_cutoff(table, train_years):
    """The 25 C default never occurs in the national (all-district) mean -- max is
    ~27 C but only in pre-monsoon weeks with RH < 70 -- so the feature would be all
    zeros. Re-tune on TRAINING seasons only: 75th percentile of weekly mean temp."""
    return float(table.loc[table.year.isin(train_years), "tmean"].quantile(0.75))


def build_features(table, warm_c=WARM_C, wet_rh=WET_RH):
    """table: weekly_table sorted by (year, week). Returns a copy with feature columns."""
    t = table.sort_values(["year", "week"]).reset_index(drop=True).copy()
    rate = t.cases                                           # rate = cases (no site counts)
    lr = np.log1p(rate)
    for k in range(4):
        t[f"log_cases_{k}"] = lr.shift(k)
    t["growth_2"] = lr - lr.shift(2)

    t["tmin_mean_4"] = t.tmin.rolling(4, min_periods=4).mean()
    t["tmin_mean_8"] = t.tmin.rolling(8, min_periods=8).mean()
    t["tmean_mean_lag4_12"] = t.tmean.shift(4).rolling(9, min_periods=9).mean()   # weeks t-12..t-4
    t["heat_above_18_8"] = (t.tmean - 18).clip(lower=0).rolling(8, min_periods=8).sum()
    t["rain_sum_4"] = t.rain.rolling(4, min_periods=4).sum()
    t["rain_sum_8"] = t.rain.rolling(8, min_periods=8).sum()
    t["rh_mean_4"] = t.rh.rolling(4, min_periods=4).mean()
    ww = ((t.tmean >= warm_c) & (t.rh >= wet_rh)).astype(float).where(t.tmean.notna() & t.rh.notna())
    t["warm_wet_weeks_8"] = ww.rolling(8, min_periods=8).sum()

    # previous calendar year's total (immunity stand-in); NaN if that year has < 40 weeks of data
    yearly = t.groupby("year").cases.agg(["sum", "count"])
    yearly["total"] = yearly["sum"].where(yearly["count"] >= 40)
    t["log_last_season"] = np.log1p(t.year.map(lambda y: yearly.total.get(y - 1, np.nan)))

    w = np.minimum(t.week, 52)
    t["woy_sin"] = np.sin(2 * np.pi * w / 52)
    t["woy_cos"] = np.cos(2 * np.pi * w / 52)
    return t
