import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN

# =========================
# ✅ SESSION STATE INIT (FIX)
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
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# 🎨 UI STYLE
# =========================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #e3f2fd, #ffffff);}
h1 {text-align: center;color: #0d47a1;font-weight: bold;}
h3 {text-align: center;color: #1565c0;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #1976d2, #42a5f5);}
section[data-testid="stSidebar"] h1, h2, h3 {color: white;}
.sidebar-steps {background: white;padding: 15px;border-radius: 15px;color: black !important;}
.stButton > button {background: linear-gradient(to right, #1976d2, #42a5f5);color: white;border-radius: 12px;}
.card {padding: 18px;border-radius: 15px;background: #f5faff;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 ML Pipeline Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3>📊 End-to-End Machine Learning Automation System</h3>", unsafe_allow_html=True)

# =========================
# STEPS
# =========================
steps = [
    "1. Problem Type","2. Dataset","3. EDA","4. Cleaning",
    "5. Feature Selection","6. Split","7. Model","8. Training","9. Metrics"
]

with st.sidebar:
    st.markdown("### ⚙️ Steps")
    st.markdown("<div class='sidebar-steps'>", unsafe_allow_html=True)
    step = st.radio("", steps)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# STEP 1
# =========================
if step == steps[0]:
    st.session_state.problem = st.radio("Select Problem Type",
                                       ["Classification", "Regression", "Clustering"])

# =========================
# STEP 2
# =========================
elif step == steps[1]:
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("Location_Wise_Student_Data.csv")

    st.session_state.data = df
    st.session_state.target = st.selectbox("Select Target Column", df.columns)

    st.dataframe(df.head())

# =========================
# STEP 3
# =========================
elif step == steps[2]:

    if st.session_state.data is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    df = st.session_state.data
    st.dataframe(df.head())
    st.write(df.describe())

# =========================
# STEP 4
# =========================
elif step == steps[3]:

    if st.session_state.data is None:
        st.warning("⚠️ Upload dataset first")
        st.stop()

    df = st.session_state.data.copy()

    if st.checkbox("Fill Missing (Mean)"):
        df = df.fillna(df.mean(numeric_only=True))

    st.session_state.data = df
    st.write(df.shape)

# =========================
# STEP 5
# =========================
elif step == steps[4]:

    if st.session_state.data is None or st.session_state.target is None:
        st.warning("⚠️ Complete Step 2 first")
        st.stop()

    df = st.session_state.data
    target = st.session_state.target

    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    X = pd.get_dummies(X).fillna(0)
    y = y.astype("category").cat.codes

    method = st.selectbox("Method", ["None", "Variance", "ANOVA"])

    if method == "Variance":
        selector = VarianceThreshold(0.0)
        try:
            X = selector.fit_transform(X)
        except:
            st.warning("Variance issue")

    elif method == "ANOVA":
        X = SelectKBest(f_classif, k="all").fit_transform(X, y)

    st.session_state.X = X
    st.session_state.y = y

    st.write("Shape:", X.shape)

# =========================
# STEP 6
# =========================
elif step == steps[5]:

    if st.session_state.X is None:
        st.warning("⚠️ Do Feature Selection first")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.X, st.session_state.y, test_size=0.2)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split done")

# =========================
# STEP 7
# =========================
elif step == steps[6]:

    if st.session_state.problem is None:
        st.warning("⚠️ Select problem type first")
        st.stop()

    if st.session_state.problem == "Regression":
        model = "Linear"
    else:
        model = st.selectbox("Model", ["Logistic", "SVM", "RandomForest"])

    st.session_state.model_name = model

# =========================
# STEP 8
# =========================
elif step == steps[7]:

    if st.session_state.X_train is None:
        st.warning("⚠️ Split data first")
        st.stop()

    model_name = st.session_state.model_name

    if model_name == "Logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "RandomForest":
        model = RandomForestClassifier()
    else:
        model = LinearRegression()

    if st.button("Train"):

        model.fit(st.session_state.X_train, st.session_state.y_train)

        # ✅ CV FIX
        y_train = st.session_state.y_train
        k = 3

        if len(np.unique(y_train)) < k:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=k)
        else:
            cv = StratifiedKFold(n_splits=k)

        scores = cross_val_score(model, st.session_state.X_train, y_train, cv=cv)

        st.success(f"Score: {scores.mean():.4f}")

        st.session_state.model = model

# =========================
# STEP 9
# =========================
elif step == steps[8]:

    if st.session_state.model is None:
        st.warning("⚠️ Train model first")
        st.stop()

    preds = st.session_state.model.predict(st.session_state.X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(st.session_state.y_test, preds)

    st.success(f"Accuracy: {acc:.2f}")
