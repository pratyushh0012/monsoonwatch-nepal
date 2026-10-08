"""
MonsoonWatch Nepal -- interactive demo.

    streamlit run app/app.py

Loads the saved model (models/dengue/denguewatch.joblib) and the pre-computed
back-test; nothing is retrained on start-up.
The earlier, detailed version is in archive/denguewatch_app_detailed.py.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # the saved model pickles src.models classes

BLUE, ORANGE, GREY, GRID, TEAL = "#2a78d6", "#eb6834", "#6b7570", "#e4e3df", "#0f7c74"
LIGHTS = {
    "green": {"name": "Normal", "colour": "#0ca30c", "advice": "No surge expected in the next 4 weeks."},
    "amber": {"name": "Get ready", "colour": "#e8a200", "advice": "A surge could start soon. Check beds, platelets and staff."},
    "red": {"name": "Stand by", "colour": "#d03b3b", "advice": "A big surge is likely. Start surge plans."},
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="MonsoonWatch Nepal", layout="centered")
st.markdown("""<style>
.block-container {max-width: 900px; padding-top: 2rem;}
.light {display:flex; align-items:center; gap:14px; padding:14px 18px; border-radius:14px; margin:4px 0 10px 0;}
.light .dot {width:34px; height:34px; border-radius:50%; flex:none;}
.light b {font-size:1.6rem;}
.light span {font-size:1.05rem; opacity:.85;}
.dots {display:flex; gap:10px; margin:6px 0 4px 0;}
.dots div {width:44px; height:44px; border-radius:50%; display:grid; place-items:center; color:#fff; font-weight:700; font-size:1.2rem;}
.sub {font-size:1.1rem; opacity:.8; margin-top:-8px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ROOT, "models/dengue/denguewatch.joblib"))


@st.cache_data
def load_data():
    table = pd.read_csv(os.path.join(ROOT, "data/processed/weekly_table.csv"), parse_dates=["week_end"])
    preds = pd.read_csv(os.path.join(ROOT, "data/processed/backtest_predictions.csv"))
    return table, preds


B = load_model()
table, preds = load_data()
TB = table.reset_index(drop=True)
TH = B["thresholds"]
LINE = B["surge_line"].set_index("woy").line
REG, CLS = B["chosen_reg"], B["chosen_cls"]


# ------------------------------------------------------------------ helpers
def traffic_light(prob, p90, week):
    line4 = LINE[(min(int(week), 52) + 4 - 1) % 52 + 1]
    if prob >= TH["amber"] and p90 > TH["red_p90_x_line"] * line4:
        return "red"
    return "amber" if prob >= TH["green"] else "green"


def nice(n):
    n = float(n)
    if n < 20:
        return f"{n:.0f}"
    if n < 200:
        return f"{round(n / 5) * 5:.0f}"
    if n < 2000:
        return f"{round(n / 10) * 10:,.0f}"
    return f"{round(n / 100) * 100:,.0f}"


def light_card(key):
    L = LIGHTS[key]
    st.markdown(f"<div class='light' style='background:{L['colour']}22'>"
                f"<div class='dot' style='background:{L['colour']}'></div>"
                f"<div><b>{L['name']}</b><br><span>{L['advice']}</span></div></div>", unsafe_allow_html=True)


def gauge(prob, height=230):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Chance a surge starts in the next 4 weeks", "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 30, 60, 100]},
               "bar": {"color": "#18211e", "thickness": 0.25},
               "steps": [{"range": [0, 30], "color": LIGHTS["green"]["colour"]},
                         {"range": [30, 60], "color": LIGHTS["amber"]["colour"]},
                         {"range": [60, 100], "color": LIGHTS["red"]["colour"]}]}))
    fig.update_layout(height=height, margin=dict(l=30, r=30, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def style(fig, height=340, ytitle="Cases per week"):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", font=dict(size=14),
                      legend=dict(orientation="h", y=1.12, x=0), yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, rangemode="tozero")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def predict(cases, week, last_year_total):
    """Model inputs from a few answers; weather set to normal for that time of year."""
    w = min(week, 52)
    clim = table[table.year >= 2017].assign(woy=np.minimum(table.week, 52)).groupby("woy")[["tmean", "tmin", "rain", "rh"]].mean()
    past = lambda n: [((w - 1 - k) % 52) + 1 for k in range(n)]
    tm8, rh8 = clim.tmean.loc[past(8)], clim.rh.loc[past(8)]
    x = pd.DataFrame([{
        "log_cases_0": np.log1p(cases), "log_cases_1": np.log1p(cases), "log_cases_2": np.log1p(cases),
        "log_cases_3": np.log1p(cases), "growth_2": 0.0, "log_last_season": np.log1p(last_year_total),
        "woy_sin": np.sin(2 * np.pi * w / 52), "woy_cos": np.cos(2 * np.pi * w / 52),
        "tmin_mean_4": clim.tmin.loc[past(4)].mean(), "tmin_mean_8": clim.tmin.loc[past(8)].mean(),
        "tmean_mean_lag4_12": clim.tmean.loc[[((w - 1 - k) % 52) + 1 for k in range(4, 13)]].mean(),
        "heat_above_18_8": (tm8 - 18).clip(lower=0).sum(),
        "rain_sum_4": clim.rain.loc[past(4)].sum(), "rain_sum_8": clim.rain.loc[past(8)].sum(),
        "rh_mean_4": clim.rh.loc[past(4)].mean(),
        "warm_wet_weeks_8": float(((tm8.values >= B["warm_cutoff"]) & (rh8.values >= 70)).sum())}])
    for c, (lo, hi) in B["feature_range"].items():
        x[c] = x[c].clip(lo, hi)
    return float(B["models"][CLS].predict(x).prob.iloc[0]), B["models"][REG].predict(x).iloc[0]


# back-test: one row per test week, with the official warning + case forecast
p = preds[preds.line_method == B["line_method"]]
cls = p[(p.model + "_" + p.set) == CLS][["row", "year", "week", "prob", "cases", "onset"]]
reg = p[(p.model + "_" + p.set) == REG][["row", "mean", "p90", "y_reg"]]
BT = cls.merge(reg, on="row", how="left")
BT["date"] = TB.week_end.values[BT.row.values]
BT["light"] = [traffic_light(pr, q if pd.notna(q) else 0, w) for pr, q, w in zip(BT.prob, BT.p90, BT.week)]
BT = BT[BT.row + 4 < len(TB)]
BT["target_date"] = TB.week_end.values[BT.row.values + 4]

# ================================================================== header
st.title("MonsoonWatch Nepal")
st.markdown("<p class='sub'>A 4-week early warning for dengue surges, from Nepal's weekly hospital reports.</p>",
            unsafe_allow_html=True)

t_now, t_hist, t_replay, t_try, t_about = st.tabs(["Now", "History", "Replay a year", "Try it", "About"])

# ================================================================== now
with t_now:
    lt = B["latest"]
    row = table[(table.year == lt["year"]) & (table.week == lt["week"])].iloc[0]
    light_card(traffic_light(lt["prob"], lt["p90"], lt["week"]))
    a, b = st.columns([3, 2])
    with a:
        gauge(lt["prob"])
    with b:
        st.metric("Expected cases per week, 4 weeks ahead", f"about {nice(lt['mean'])}",
                  help=f"Likely range {nice(lt['p10'])} to {nice(lt['p90'])}")
        st.metric("Cases reported in the latest week", f"{row.cases:.0f}")
        st.caption(f"Latest report: week ending {row.week_end:%d %b %Y}")

# ================================================================== history
with t_hist:
    yrs = sorted(table.year[table.year >= 2018].unique())
    pick = st.radio("Show", ["All years"] + [str(y) for y in yrs], horizontal=True, key="hist_years")
    t = table[table.year >= 2018].copy() if pick in (None, "All years") else table[table.year == int(pick)].copy()
    t["usual"] = np.minimum(t.week, 52).map(LINE)
    above = (t.cases > t.usual).fillna(False).values
    fig = go.Figure()
    d = t.week_end.reset_index(drop=True)
    starts = np.where(above & ~np.r_[False, above[:-1]])[0]
    ends = np.where(above & ~np.r_[above[1:], False])[0]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=d[s] - pd.Timedelta(days=3.5), x1=d[e] + pd.Timedelta(days=3.5),
                      fillcolor=ORANGE, opacity=0.15, line_width=0)
    fig.add_trace(go.Scatter(x=t.week_end, y=t.cases, name="Cases each week", line=dict(color=BLUE, width=2.5),
                             hovertemplate="%{y:,.0f} cases"))
    fig.add_trace(go.Scatter(x=t.week_end, y=t.usual, name="Usual for that time of year",
                             line=dict(color=GREY, width=1.5, dash="dash"), hovertemplate="usual %{y:,.0f}"))
    show(style(fig, 380))
    st.caption("Shaded = surge weeks (above the usual level). Hover to see each week.")

# ================================================================== replay
with t_replay:
    surges, caught, leads = 0, 0, []
    for _, g in BT.groupby("year"):
        g = g.set_index("row")
        alert = g.prob >= TH["green"]
        for s in g.index[g.onset == 1]:
            surges += 1
            win = [r for r in range(s - 4, s) if r in alert.index and alert[r]]
            if win:
                caught += 1
                leads.append(s - min(win))
    c1, c2 = st.columns([3, 2])
    with c1:
        dots = "".join(f"<div style='background:{'#0ca30c' if i < caught else '#d03b3b'}'>{'✓' if i < caught else '✗'}</div>"
                       for i in range(surges))
        st.markdown(f"<div class='dots'>{dots}</div>", unsafe_allow_html=True)
        st.caption(f"Past surges warned about: {caught} of {surges}")
    with c2:
        st.metric("Warning given", f"~{np.mean(leads):.0f} weeks early")

    yrs_bt = sorted(int(y) for y in BT.year.unique())
    year = st.radio("Year", yrs_bt, index=yrs_bt.index(2024), horizontal=True, key="replay_year")
    g = BT[BT.year == year].sort_values("date").reset_index(drop=True)
    labels = [f"{d:%d %b}" for d in g.date]
    pick_wk = st.select_slider("Drag through the year", options=labels, value=labels[min(30, len(g) - 1)],
                               key=f"replay_week_{year}")
    now = g.iloc[labels.index(pick_wk)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.date, y=g.cases, name="Real cases", line=dict(color=BLUE, width=2.5),
                             hovertemplate="real %{y:,.0f}"))
    fig.add_trace(go.Scatter(x=g.target_date, y=g["mean"], name="What it predicted 4 weeks before",
                             line=dict(color=ORANGE, width=2.5, dash="dot"), hovertemplate="predicted %{y:,.0f}"))
    ymax = float(np.nanmax([g.cases.max(), g["mean"].max()]))
    fig.add_trace(go.Scatter(x=g.date, y=[-0.06 * ymax] * len(g), mode="markers", name="Warning that week",
                             marker=dict(color=[LIGHTS[k]["colour"] for k in g.light], size=11, symbol="square"),
                             hovertemplate="%{text}", text=[LIGHTS[k]["name"] for k in g.light]))
    fig.add_vline(x=now.date, line=dict(color="#18211e", width=2))
    show(style(fig, 360))

    a, b = st.columns(2)
    with a:
        st.markdown(f"**On {now.date:%d %b %Y} it said:**")
        light_card(now.light)
        st.caption(f"Chance of a surge: {now.prob:.0%}")
    with b:
        st.markdown(f"**4 weeks later ({now.target_date:%d %b}):**")
        st.metric("Real cases", f"{now.y_reg:,.0f}" if pd.notna(now.y_reg) else "–",
                  delta=f"predicted {nice(now['mean'])}", delta_color="off")

    st.divider()
    st.markdown("**Simple model vs complex model**")
    metric = st.radio("Compare", ["Surges caught (of 6)", "False alarms", "Average error"], horizontal=True,
                      key="cmp")
    vals = {"Surges caught (of 6)": [4, 3, 2], "False alarms": [2, 8, 5], "Average error": [144, 202, 216]}[metric or "Surges caught (of 6)"]
    fig = go.Figure(go.Bar(x=["Simple (used here)", "Complex", "Complex + weather"], y=vals, text=vals,
                           textposition="outside", marker_color=[TEAL, "#a9b4af", "#a9b4af"]))
    fig.update_yaxes(range=[0, max(vals) * 1.25], visible=False)
    show(style(fig, 260, ""))
    st.caption("Higher is better for surges caught; lower is better for false alarms and error (cases per week).")

# ================================================================== try it
with t_try:
    a, b = st.columns([2, 3])
    with a:
        month = st.select_slider("Month", MONTHS, value="Aug", key="try_month")
        cases = st.slider("Cases reported this week", 0, 2500, 300, step=25, key="try_cases")
        last = st.radio("Last year was", ["Quiet", "Average", "Big outbreak"], index=1, horizontal=True,
                        key="try_last")
    last_total = {"Quiet": 500, "Average": 6000, "Big outbreak": 25000}[last or "Average"]
    week = round((MONTHS.index(month) + 0.5) * 52 / 12)
    prob, rr = predict(cases, week, last_total)
    usual = LINE[min(week, 52)]
    with b:
        gauge(prob, 220)
        light_card(traffic_light(prob, rr.p90, week))
    m1, m2 = st.columns(2)
    m1.metric(f"Usual for {month}", f"{nice(usual)} / week")
    m2.metric("Expected in 4 weeks", f"{nice(rr['mean'])} / week")
    if rr["mean"] > 1.5 * table.cases.max():
        st.warning("These answers are far outside anything seen before, so this number is not reliable.")

# ================================================================== about
with t_about:
    st.markdown("""
- **Data:** 482 weekly hospital reports from Nepal's health office (2016–2025), plus NASA weather.
- **Looks at:** cases right now and the time of year.
- **Tested fairly:** it only learned from earlier years, then predicted later ones.
- **Weather and complex AI** didn't make it better.
- **Limits:** whole country only, and only 5 years of testing.
""")
