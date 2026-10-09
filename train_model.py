"""Re-creates the preferred model from the CreditLens notebook (tuned Random Forest) and saves
everything the Streamlit app needs. Usage:  python train_model.py "path/to/default of credit card clients.xls"
All settings (seed 42, 80/20 stratified split, tuned parameters, 5-fold out-of-fold threshold) are the same as in the notebook."""
import sys, json
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve)
from sklearn.inspection import permutation_importance

RS = 42
path = sys.argv[1] if len(sys.argv) > 1 else "default of credit card clients.xls"

df = pd.read_excel(path, header=1).rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
X = df.drop(columns=["ID", "default", "SEX", "MARRIAGE"])
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)

cat = ["EDUCATION"]
num = [c for c in X.columns if c not in cat]
prep = ColumnTransformer([("num", StandardScaler(), num),
                          ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, max_features=0.3,
                               random_state=RS, n_jobs=-1)
pipe = Pipeline([("prep", prep), ("model", model)]).fit(X_train, y_train)

# threshold chosen on out-of-fold TRAINING predictions (as in the notebook)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
oof = cross_val_predict(clone(pipe), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
p, r, t = precision_recall_curve(y_train, oof)
f1c = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
threshold = float(t[np.argmax(f1c)])

prob = pipe.predict_proba(X_test)[:, 1]
pred = (prob >= threshold).astype(int)
metrics = dict(threshold=round(threshold, 3), accuracy=accuracy_score(y_test, pred),
               precision=precision_score(y_test, pred), recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
               roc_auc=roc_auc_score(y_test, prob), avg_precision=average_precision_score(y_test, prob),
               brier=brier_score_loss(y_test, prob), test_default_rate=float(y_test.mean()), test_size=int(len(y_test)))
print({k: round(v, 4) for k, v in metrics.items()})

# empirical default rate by probability band on the held-out test set
edges = [0, 0.15, threshold, 0.5, 1.0001]
names = ["Low", "Medium", "High", "Very high"]
bands = pd.cut(prob, bins=edges, labels=names, right=False)
band_info = {}
for n in names:
    m = (bands == n)
    band_info[n] = dict(customers=int(m.sum()), default_rate=float(y_test[m].mean()) if m.sum() else None)

# decile table
rk = pd.DataFrame({"prob": prob, "actual": y_test.values})
rk["decile"] = pd.qcut(rk["prob"].rank(method="first", ascending=False), 10, labels=range(1, 11))
dec = rk.groupby("decile", observed=True).agg(customers=("actual", "size"), defaulters=("actual", "sum"), default_rate=("actual", "mean"))
dec["cumulative_share_of_defaulters"] = (dec["defaulters"] / dec["defaulters"].sum()).cumsum()
dec.reset_index().to_csv("model/deciles.csv", index=False)

# permutation importance (test set, descriptive only)
perm = permutation_importance(pipe, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=RS, n_jobs=1)
pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean, "std": perm.importances_std}) \
  .sort_values("importance", ascending=False).to_csv("model/importance.csv", index=False)

# demo file: 200 genuinely unseen test customers
demo = X_test.sample(200, random_state=RS).copy()
demo.insert(0, "ID", df.loc[demo.index, "ID"])
demo["actual_default"] = y_test.loc[demo.index]
demo.to_csv("data/sample_customers.csv", index=False)

joblib.dump(pipe, "model/credit_model.joblib", compress=3)
json.dump(dict(threshold=threshold, features=list(X.columns), metrics=metrics, bands=band_info,
               band_edges=edges, band_names=names), open("model/config.json", "w"), indent=2)
print("saved. bands:", band_info)
