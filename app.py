"""CreditLens - Credit Risk Analytics and Default Prediction (academic portfolio project)."""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditLens", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------------------------
# Load the saved model and settings (created by train_model.py)
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/credit_model.joblib")

@st.cache_data
def load_config():
    return json.load(open("model/config.json"))

pipe = load_model()
cfg = load_config()
FEATURES = cfg["features"]
THRESHOLD = cfg["threshold"]
EDGES, BAND_NAMES = cfg["band_edges"], cfg["band_names"]
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]          # month 1 = Sep 2005 ... month 6 = Apr 2005

def band_for(p):
    for name, lo, hi in zip(BAND_NAMES, EDGES[:-1], EDGES[1:]):
        if lo <= p < hi:
            return name
    return BAND_NAMES[-1]

def band_rate_text(name):
    info = cfg["bands"][name]
    return f"{info['default_rate']*100:.0f}% of the {info['customers']:,} held-out test customers in this band actually defaulted"

def prepare(df):
    """Make an uploaded table match the training format."""
    df = df.rename(columns={"PAY_0": "PAY_1"}).copy()
    if "EDUCATION" in df:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    return df

# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
st.sidebar.title("📊 CreditLens")
page = st.sidebar.radio("Go to", ["Score one customer", "Score a portfolio (CSV)", "About the model"])
st.sidebar.markdown("---")
st.sidebar.warning("**Academic portfolio project.** Not validated or approved for any real lending decision.")

# ==============================================================================================
# PAGE 1: single customer
# ==============================================================================================
PRESETS = {
    "Custom": None,
    "Typical on-time payer": dict(LIMIT_BAL=200000, EDUCATION=2, AGE=38, PAY=[0, 0, 0, 0, 0, 0],
                                  BILL=[30000, 28000, 26000, 25000, 24000, 22000], PAYAMT=[3000, 3000, 2500, 2500, 2000, 2000]),
    "Recently delinquent": dict(LIMIT_BAL=50000, EDUCATION=3, AGE=29, PAY=[2, 2, 1, 0, 0, 0],
                                BILL=[48000, 47000, 45000, 40000, 38000, 35000], PAYAMT=[0, 0, 1500, 1500, 1500, 1500]),
}

def apply_preset():
    p = PRESETS[st.session_state["preset"]]
    if p is None:
        return
    st.session_state["LIMIT_BAL"], st.session_state["EDUCATION"], st.session_state["AGE"] = p["LIMIT_BAL"], p["EDUCATION"], p["AGE"]
    for i in range(6):
        st.session_state[f"PAY_{i+1}"] = p["PAY"][i]
        st.session_state[f"BILL_AMT{i+1}"] = p["BILL"][i]
        st.session_state[f"PAY_AMT{i+1}"] = p["PAYAMT"][i]

if page == "Score one customer":
    st.title("Score one customer")
    st.caption("Enter a customer's profile and six-month history to get an estimated probability of default next month.")

    defaults = {"LIMIT_BAL": 100000, "EDUCATION": 2, "AGE": 35}
    for i in range(1, 7):
        defaults.update({f"PAY_{i}": 0, f"BILL_AMT{i}": 20000, f"PAY_AMT{i}": 2000})
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    st.selectbox("Start from an example", list(PRESETS), key="preset", on_change=apply_preset)

    c1, c2, c3 = st.columns(3)
    c1.number_input("Credit limit (NT$)", min_value=10000, max_value=1000000, step=10000, key="LIMIT_BAL")
    c2.selectbox("Education", [1, 2, 3, 4], key="EDUCATION",
                 format_func=lambda x: {1: "1 - Graduate school", 2: "2 - University", 3: "3 - High school", 4: "4 - Others"}[x])
    c3.number_input("Age", min_value=18, max_value=100, step=1, key="AGE")

    st.subheader("Repayment status by month")
    st.caption("-2/-1/0 = no delay (-1 = paid duly); 1 = one month late; 2 = two months late; ... up to 8.")
    cols = st.columns(6)
    for i in range(6):
        cols[i].selectbox(f"{MONTHS[i]} 2005", list(range(-2, 9)), key=f"PAY_{i+1}")

    st.subheader("Bill statement amounts (NT$)")
    cols = st.columns(6)
    for i in range(6):
        cols[i].number_input(f"{MONTHS[i]} bill", step=1000, key=f"BILL_AMT{i+1}")

    st.subheader("Amounts paid (NT$)")
    cols = st.columns(6)
    for i in range(6):
        cols[i].number_input(f"{MONTHS[i]} paid", min_value=0, step=500, key=f"PAY_AMT{i+1}")

    row = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])[FEATURES]
    prob = float(pipe.predict_proba(row)[0, 1])
    flagged = prob >= THRESHOLD
    band = band_for(prob)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Estimated default probability", f"{prob*100:.1f}%")
    r2.metric("Risk band", band)
    r3.metric(f"Flagged (threshold {THRESHOLD*100:.1f}%)", "Yes" if flagged else "No")
    st.progress(min(prob, 1.0))
    st.info(f"**How to read this:** in held-out test data, {band_rate_text(band)}. "
            f"The overall default rate in that data was {cfg['metrics']['test_default_rate']*100:.1f}%.")
    st.caption("This is a statistical estimate from 2005 Taiwanese card data, shown for learning purposes. It is not a credit decision.")

# ==============================================================================================
# PAGE 2: portfolio
# ==============================================================================================
elif page == "Score a portfolio (CSV)":
    st.title("Score a portfolio")
    st.caption("Upload a CSV of customers to rank them by predicted default risk.")
    st.markdown("**Required columns:** " + ", ".join(f"`{f}`" for f in FEATURES) +
                ". Optional: `ID`, and `actual_default` or `default` (to check how the flags performed). "
                "The original UCI column name `PAY_0` is also accepted.")

    up = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.button("Or use the built-in sample (200 unseen test customers)")
    data = None
    if up is not None:
        data = pd.read_csv(up)
    elif use_sample or st.session_state.get("sample_loaded"):
        st.session_state["sample_loaded"] = True
        data = pd.read_csv("data/sample_customers.csv")

    if data is not None:
        data = prepare(data)
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif data[FEATURES].isnull().any().any():
            st.error("The required columns contain missing values. Please fill or remove those rows.")
        else:
            out = data.copy()
            out["default_probability"] = pipe.predict_proba(out[FEATURES])[:, 1]
            out["flagged"] = out["default_probability"] >= THRESHOLD
            out["risk_band"] = out["default_probability"].apply(band_for)
            out = out.sort_values("default_probability", ascending=False).reset_index(drop=True)
            out.insert(0, "rank", out.index + 1)

            m1, m2, m3 = st.columns(3)
            m1.metric("Customers scored", f"{len(out):,}")
            m2.metric("Flagged as high risk", f"{int(out['flagged'].sum()):,} ({out['flagged'].mean()*100:.1f}%)")
            m3.metric("Average predicted probability", f"{out['default_probability'].mean()*100:.1f}%")

            st.subheader("Customers by risk band")
            st.bar_chart(out["risk_band"].value_counts().reindex(BAND_NAMES).fillna(0))

            actual_col = "actual_default" if "actual_default" in out else ("default" if "default" in out else None)
            if actual_col:
                tp = int(((out["flagged"]) & (out[actual_col] == 1)).sum())
                st.subheader("Reality check (actual outcomes were provided)")
                a, b = st.columns(2)
                a.metric("Defaulters caught by the flag", f"{tp / max((out[actual_col] == 1).sum(), 1) * 100:.0f}%")
                b.metric("Flagged customers who really defaulted", f"{tp / max(int(out['flagged'].sum()), 1) * 100:.0f}%")
                st.caption("These are recall and precision on this file. With only a few hundred rows they are noisy.")

            st.subheader("Ranked customers (highest risk first)")
            show = ["rank"] + (["ID"] if "ID" in out else []) + ["default_probability", "risk_band", "flagged"] + \
                   ([actual_col] if actual_col else []) + ["LIMIT_BAL", "PAY_1", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
            st.dataframe(out[show].style.format({"default_probability": "{:.1%}"}), width="stretch", height=400)
            st.download_button("Download full scored table (CSV)", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")

# ==============================================================================================
# PAGE 3: about
# ==============================================================================================
else:
    st.title("About the model")
    m = cfg["metrics"]
    st.markdown(
        "**CreditLens** predicts whether a credit-card customer will **default next month**, using the UCI *Default of Credit Card Clients* "
        "dataset (30,000 customers, Taiwan, April-September 2005; Yeh & Lien, 2009). The deployed model is a **tuned Random Forest** "
        "chosen in the project notebook by cross-validated average precision on the training data.")

    st.subheader("Performance on the held-out test set (6,000 customers)")
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[1].metric("Avg precision", f"{m['avg_precision']:.3f}")
    c[2].metric("Precision", f"{m['precision']:.3f}")
    c[3].metric("Recall", f"{m['recall']:.3f}")
    c[4].metric("F1", f"{m['f1']:.3f}")
    st.caption(f"Decision threshold {THRESHOLD:.3f}, chosen to maximise F1 on out-of-fold *training* predictions, not on the test set. "
               "Roughly half of the customers flagged at this threshold are false alarms, so the model suits ranking and prioritising better than automatic decisions.")

    st.subheader("All models compared (test set)")
    st.dataframe(pd.read_csv("model/model_comparison.csv"), width="stretch", hide_index=True)
    st.caption("The four tree ensembles and the neural network are statistically indistinguishable (bootstrap intervals overlap), "
               "so the choice among them is not strongly supported by the data.")

    st.subheader("What drives the predictions (permutation importance)")
    imp = pd.read_csv("model/importance.csv").head(12).set_index("feature")
    st.bar_chart(imp["importance"])
    st.caption("Importance = drop in average precision when a feature is shuffled. Recent repayment status (PAY_1) dominates. "
               "This shows predictive importance, not cause and effect.")

    st.subheader("Ranking power (test set)")
    dec = pd.read_csv("model/deciles.csv")
    st.dataframe(dec.style.format({"default_rate": "{:.1%}", "cumulative_share_of_defaulters": "{:.1%}"}), width="stretch", hide_index=True)
    st.caption("Decile 1 = the 10% of customers with the highest predicted risk.")

    st.subheader("Limitations")
    st.markdown(
        "- One dataset, one market, one short period (2005); one-month default label; no rejected applicants.\n"
        "- Some codes in the data are undocumented (repayment status -2 and 0).\n"
        "- `SEX` and `MARRIAGE` were excluded from the model, but fairness testing would still be needed because other variables can act as proxies.\n"
        "- The threshold ignores real-world costs of missed defaulters versus false alarms.\n"
        "- **Not validated for lending.** Real use would need out-of-time validation, calibration checks, monitoring, fairness assessment and governance.")
