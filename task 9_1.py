# Task 9.1 as one PyCharm script: save the model, write the Streamlit app, test it
# Install once in the PyCharm terminal:  pip install streamlit scikit-learn joblib pandas
import os
RS = 42  # random_state used everywhere
PROJECT = r"C:\users\joram\OneDrive\Desktop\CSC4110_G06"
for d in ["data", "models", "outputs", "app", "report", "tests"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
os.chdir(PROJECT)  # every path below is relative to the project folder


# Data and the eight inputs the app will collect
import json, datetime, joblib, sklearn
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("data/stage1_flights.csv")
FEATURES = ["airline", "source_city", "destination_city", "stops", "departure_time", "class", "days_left", "duration"]
X, y = df[FEATURES], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)  # same split rule in Task 9.2
print(X_train.shape, X_test.shape)

# One full pipeline = preprocessing + model, so raw app inputs can go straight in
times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
def build(model):
    pre = ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="ignore"), ["airline", "source_city", "destination_city", "class"]),  # unordered
        ("ord", OrdinalEncoder(categories=[["zero", "one", "two_or_more"], times]), ["stops", "departure_time"]),  # ordered
        ("num", StandardScaler(), ["days_left", "duration"]),  # numeric
    ])
    return Pipeline([("prep", pre), ("model", model)])
candidates = {
    "Ridge regression": build(Ridge(alpha=1.0)),
    "Random forest": build(RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=RS, n_jobs=-1)),
    "Hist gradient boosting": build(HistGradientBoostingRegressor(random_state=RS)),
}
cmp = {}
for name, p in candidates.items():
    p.fit(X_train, y_train)
    cmp[name] = {"MAE": mean_absolute_error(y_test, p.predict(X_test)), "RMSE": float(np.sqrt(mean_squared_error(y_test, p.predict(X_test)))), "R2": r2_score(y_test, p.predict(X_test))}
cmp = pd.DataFrame(cmp).T; print(cmp.round(3))
best_name = cmp["RMSE"].idxmin()
final_pipe = candidates[best_name]  # the best fare regressor, already fitted on the training rows
print("Final model:", best_name)

# Save the pipeline with joblib.dump, plus a small information file used by the app
joblib.dump(final_pipe, "models/G06_model.joblib", compress=3)
info = {"model_name": best_name, "test_rmse": float(cmp.loc[best_name, "RMSE"]), "test_mae": float(cmp.loc[best_name, "MAE"]),
        "test_r2": float(cmp.loc[best_name, "R2"]), "trained_on": "Kaggle Flight Price Prediction (Clean_Dataset.csv, 20,000-row sample)",
        "trained_date": str(datetime.date.today()), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "choices": {c: sorted(df[c].unique().tolist()) for c in ["airline", "source_city", "destination_city", "class"]}
                   | {"stops": ["zero", "one", "two_or_more"], "departure_time": times}}
json.dump(info, open("models/G06_model_info.json", "w"), indent=2)
print("Saved. Size (MB):", round(os.path.getsize("models/G06_model.joblib") / 1e6, 2))

# Reload in a new step and predict one test row to prove it works
loaded = joblib.load("models/G06_model.joblib")
row = X_test.iloc[[0]]
print(row.to_dict("records")[0])
print("Predicted fare:", round(float(loaded.predict(row)[0])), "| actual fare:", int(y_test.iloc[0]))

app_code = """import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # project folder (works locally and on Streamlit Cloud)
MODEL_PATH = ROOT / "models" / "G06_model.joblib"
INFO_PATH = ROOT / "models" / "G06_model_info.json"

st.set_page_config(page_title="Air fare estimator", page_icon=":airplane:")
st.title("Safari Fare Desk: air fare estimator")
st.markdown("**Group G06** | CSC 4110 Machine Learning, Kibabii University")
st.write("Members: Daniel Natembeya, Ledama Nomali, Raynner Kavulika, Joram Sayia, Emmanuel Wekesa, Shikuku Cyrus")
st.write("Dataset: Flight Price Prediction (Clean_Dataset.csv, Kaggle; 300,153 Indian domestic flight fares)")


@st.cache_resource
def load_model():
    # load the saved pipeline and its information file once
    return joblib.load(MODEL_PATH), json.loads(INFO_PATH.read_text())


model, info = load_model()
ch = info["choices"]

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", ch["airline"])
    source_city = st.selectbox("Source city", ch["source_city"])
    destination_city = st.selectbox("Destination city", ch["destination_city"], index=1)
    stops = st.selectbox("Stops", ch["stops"], index=1)
with col2:
    departure_time = st.selectbox("Departure time", ch["departure_time"], index=1)
    travel_class = st.selectbox("Class", ch["class"])
    days_left = st.slider("Days left before departure", 1, 49, 20)
    duration = st.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=10.0, step=0.25)

if st.button("Predict"):
    if source_city == destination_city:
        st.warning("Source and destination are the same city, so the estimate may be unreliable.")
    row = pd.DataFrame([{"airline": airline, "source_city": source_city, "destination_city": destination_city,
                         "stops": stops, "departure_time": departure_time, "class": travel_class,
                         "days_left": days_left, "duration": duration}])
    fare = float(model.predict(row)[0])
    rmse = info["test_rmse"]
    st.success(f"Estimated fare: {fare:,.0f} INR  (+/- {rmse:,.0f} INR)")
    st.caption(f"Likely range: {max(fare - rmse, 0):,.0f} to {fare + rmse:,.0f} INR. The range is the model's test RMSE.")

st.divider()
st.caption(f"Model: {info['model_name']}. Trained on {info['trained_on']} fare data. "
           "Fares change daily, so treat this as an estimate and confirm in the airline system.")
"""
open("app/app.py", "w", encoding="utf-8").write(app_code)  # write the Streamlit app
print("app/app.py written,", len(app_code.splitlines()), "lines")

# requirements.txt: streamlit, scikit-learn, joblib and pandas (scikit-learn pinned so the saved model loads online)
import streamlit
reqs = f"streamlit>={streamlit.__version__.split('.')[0]}.0\nscikit-learn=={sklearn.__version__}\njoblib>={joblib.__version__.split('.')[0]}.0\npandas\n"
open("app/requirements.txt", "w").write(reqs)
print(reqs)

# Test the app logic without a browser (Streamlit's AppTest), choosing sensible inputs and pressing Predict
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.abspath("app/app.py"), default_timeout=60).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
print(at.success[0].value)

# Check that the screenshot has been saved
print("Screenshot present:", os.path.exists("outputs/app_screenshot.png"))
if not os.path.exists("outputs/app_screenshot.png"):
    print("Save your screenshot as outputs/app_screenshot.png (a required output file).")

# Written answer using this run's numbers
print(f"""Our group followed these stages: (1) set up the project and collected the Kaggle dataset, (2) inspected and cleaned the data and took a 20,000-row stratified sample, (3) transformed it, explored it with charts and split it 80/20 without leakage, (4) built and compared regression models (linear, polynomial, ridge, lasso) and classifiers (logistic regression, k-NN, Naive Bayes, SVM, trees, ensembles, neural networks), (5) evaluated and tuned them with cross-validation and grid search, (6) packaged the best fare regressor ({best_name}, test RMSE {info['test_rmse']:,.0f} INR), and (7) deployed it as a Streamlit app with a model card and a smoke test. We saved the whole pipeline and not only the model because the model alone expects numbers that were produced by the encoders and scaler; the pipeline repeats exactly the same preprocessing on the raw airline, city, stops and time values a user types in, so there is no mismatch or leakage and the app needs no separate preprocessing code.""")