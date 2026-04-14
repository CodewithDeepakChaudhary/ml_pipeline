import streamlit as st

# =========================
# ✅ SESSION STATE INIT
# =========================
defaults = {
    "data": None,
    "target": None,
    "problem": None,
    "X": None,
    "y": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "model": None,
    "model_name": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score

# =========================
# UI
# =========================
st.set_page_config(layout="wide")

st.title("🚀 ML Pipeline Dashboard")
st.subheader("End-to-End Machine Learning System")

steps = [
    "1. Problem Type",
    "2. Dataset",
    "3. EDA",
    "4. Cleaning",
    "5. Feature Selection",
    "6. Split",
    "7. Model",
    "8. Training",
    "9. Metrics",
]

step = st.sidebar.radio("Steps", steps)

# =========================
# STEP 1
# =========================
if step == steps[0]:
    problem = st.radio("Select Problem Type", ["Classification", "Regression", "Clustering"])
    st.session_state.problem = problem

# =========================
# STEP 2
# =========================
elif step == steps[1]:
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.session_state.data = df
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)
        st.session_state.target = target

# =========================
# STEP 3
# =========================
elif step == steps[2]:
    df = st.session_state.get("data")

    if df is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    st.dataframe(df.head())
    st.write(df.describe())

# =========================
# STEP 4
# =========================
elif step == steps[3]:
    df = st.session_state.get("data")

    if df is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    df = df.copy()

    option = st.selectbox("Missing Value Handling", ["None", "Mean", "Median"])

    if option == "Mean":
        df = df.fillna(df.mean(numeric_only=True))

    elif option == "Median":
        df = df.fillna(df.median(numeric_only=True))

    st.session_state.data = df
    st.success("Cleaning Done ✅")

# =========================
# STEP 5
# =========================
elif step == steps[4]:
    df = st.session_state.get("data")
    target = st.session_state.get("target")

    if df is None or target is None:
        st.warning("⚠️ Complete Step 2 first")
        st.stop()

    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    X = pd.get_dummies(X).fillna(0)
    y = y.astype("category").cat.codes

    st.session_state.X = X
    st.session_state.y = y

    st.success("Feature Selection Done ✅")

# =========================
# STEP 6
# =========================
elif step == steps[5]:
    X = st.session_state.get("X")
    y = st.session_state.get("y")

    if X is None or y is None:
        st.warning("⚠️ Complete Feature Selection first")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split Done ✅")

# =========================
# STEP 7
# =========================
elif step == steps[6]:
    problem = st.session_state.get("problem")

    if problem is None:
        st.warning("⚠️ Select problem type first")
        st.stop()

    if problem == "Regression":
        model_name = st.selectbox("Model", ["Linear"])
    elif problem == "Clustering":
        model_name = st.selectbox("Model", ["KMeans"])
    else:
        model_name = st.selectbox("Model", ["Logistic", "SVM", "RandomForest"])

    st.session_state.model_name = model_name

# =========================
# STEP 8
# =========================
elif step == steps[7]:
    X_train = st.session_state.get("X_train")
    y_train = st.session_state.get("y_train")
    model_name = st.session_state.get("model_name")

    if X_train is None or y_train is None or model_name is None:
        st.warning("⚠️ Complete previous steps")
        st.stop()

    if model_name == "Logistic":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "SVM":
        model = SVC()

    elif model_name == "RandomForest":
        model = RandomForestClassifier()

    elif model_name == "Linear":
        model = LinearRegression()

    elif model_name == "KMeans":
        model = KMeans()

    if st.button("Train Model"):
        if model_name == "KMeans":
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)

        st.session_state.model = model
        st.success("Model Trained 🎉")

# =========================
# STEP 9
# =========================
elif step == steps[8]:
    model = st.session_state.get("model")
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")

    if model is None or X_test is None or y_test is None:
        st.warning("⚠️ Complete training first")
        st.stop()

    preds = model.predict(X_test)

    if st.button("Evaluate"):
        acc = accuracy_score(y_test, preds)
        st.success(f"Accuracy: {acc:.2f}")
