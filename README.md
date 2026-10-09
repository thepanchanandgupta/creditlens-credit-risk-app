# CreditLens: Credit Risk Analytics and Default Prediction

An end-to-end credit-risk project on the UCI *Default of Credit Card Clients* dataset (30,000 customers, Taiwan, 2005), with a Streamlit app for scoring customers.

> **Academic portfolio project. Not validated or approved for any real lending decision.**

## What is in this repo
| Path | Purpose |
|---|---|
| `notebook/CreditLens_Credit_Risk_Project.ipynb` | Full analysis with outputs: EDA, leakage-free preprocessing, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, tuning, a Keras neural network, comparison and interpretation |
| `train_model.py` | Rebuilds the preferred model (tuned Random Forest) exactly as in the notebook and saves it |
| `app.py` | Streamlit app: score one customer, score a CSV portfolio, model information |
| `model/` | Saved pipeline (`credit_model.joblib`), decision threshold and supporting tables |
| `data/sample_customers.csv` | 200 unseen test customers for the demo upload |

## Key results (held-out test set, 6,000 customers)
Tuned Random Forest: ROC-AUC 0.774, average precision 0.554, F1 0.545 (precision 0.511, recall 0.583) at a threshold of 0.245. The top-risk decile has a 69% default rate versus 22% overall. Tree ensembles and the neural network were statistically indistinguishable; `PAY_1` (latest repayment status) is by far the most important feature.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the model (optional)
Download `default of credit card clients.xls` from the [UCI page](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), then:
```bash
pip install xlrd
python train_model.py "default of credit card clients.xls"
```
Keep `scikit-learn` at the version in `requirements.txt`; a saved model may not load on a different version.

## Deploy on Streamlit Community Cloud
1. Sign in at https://share.streamlit.io with GitHub.
2. Create app, pick this repository, branch `main`, main file `app.py`.
3. Deploy.

## Limitations
One dataset, market and period; one-month default label; no rejected applicants; undocumented codes; `SEX` and `MARRIAGE` excluded but fairness testing would still be needed; threshold ignores real costs. Real use would require out-of-time validation, calibration, monitoring, fairness assessment and governance.

## Citation
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473-2480.
