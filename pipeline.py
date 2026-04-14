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
from sklearn.metrics import accuracy_score, mean_squared_error

# =========================
# 🎨 UI STYLE
# =========================
st.set_page_config(layout="wide")

st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
}

/* ===== MAIN TITLE ===== */
h1 {
    text-align: center;
    color: #0d47a1;
    font-weight: bold;
}

/* ===== SUBTITLE ===== */
h3 {
    text-align: center;
    color: #1565c0;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1976d2, #42a5f5);
}

/* ===== SIDEBAR TEXT VISIBLE ===== */
/* Sidebar title only white */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white;
}

/* Steps card text black */
.sidebar-steps {
    background: white;
    padding: 15px;
    border-radius: 15px;
    color: black !important;
}

/* Radio text visible */
.sidebar-steps label {
    color: black !important;
}

/* ===== SIDEBAR STEPS BOX ===== */
.sidebar-steps {
    background: white;
    padding: 15px;
    border-radius: 15px;
    color: black !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

/* ===== RADIO BUTTON TEXT ===== */
.stRadio label {
    color: black !important;
    font-weight: 500;
}

/* ===== ACTIVE STEP ===== */
.stRadio div[role="radiogroup"] label[data-checked="true"] {
    font-weight: bold;
    color: #d32f2f !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(to right, #1976d2, #42a5f5);
    color: white;
    border-radius: 12px;
    padding: 8px 20px;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(to right, #1565c0, #1e88e5);
    transform: scale(1.05);
}

/* ===== CARD UI ===== */
.card {
    padding: 18px;
    border-radius: 15px;
    background: #f5faff;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

/* ===== METRIC BOX ===== */
.metric-box {
    padding: 12px;
    border-radius: 12px;
    background: #e1f5fe;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
}

/* ===== SUCCESS ===== */
.stSuccess {
    background-color: #e8f5e9 !important;
    border-radius: 10px;
}

/* ===== ERROR ===== */
.stError {
    background-color: #ffebee !important;
    border-radius: 10px;
}

/* ===== SELECTBOX ===== */
.stSelectbox div {
    border-radius: 10px;
}

/* ===== RADIO BUTTON ===== */
.stRadio > div {
    padding: 10px;
    border-radius: 10px;
    background: #f1f8ff;
}

</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("<h1>🚀 ML Pipeline Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3>📊 End-to-End Machine Learning Automation System</h3>", unsafe_allow_html=True)
# =========================
# STEPS
# =========================
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

with st.sidebar:

    st.markdown("### ⚙️ Steps")

    st.markdown("<div class='sidebar-steps'>", unsafe_allow_html=True)

    step = st.radio("", steps)

    st.markdown("</div>", unsafe_allow_html=True)
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
    else:
        df = pd.read_csv("Location_Wise_Student_Data.csv")

    st.session_state.data = df

    target = st.selectbox("Select Target Column", df.columns)
    st.session_state.target = target

    st.dataframe(df.head())

# =========================
# STEP 3
# =========================
elif step == steps[2]:
    df = st.session_state.data

    st.markdown('<div class="card">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.dataframe(df.head())
    st.write(df.describe())

    # =========================
    # 🔥 CORRELATION HEATMAP
    # =========================
    
    st.subheader("🔥 Correlation Heatmap")
    
    # Select only numeric columns
    num_df = df.select_dtypes(include=np.number)
    
    if num_df.shape[1] < 2:
        st.warning("⚠️ Not enough numeric columns for correlation heatmap")
    else:
        corr = num_df.corr()
    
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    
        st.pyplot(fig)

# =========================
# STEP 4 CLEANING
# =========================
elif step == steps[3]:

    st.markdown("""
    <div style='padding:15px; border-radius:10px; background-color:#f0f2f6'>
        <h3>🧹 Data Cleaning</h3>
    </div>
    """, unsafe_allow_html=True)

    # ✅ MUST DEFINE df HERE
    df = st.session_state.data.copy()

    col1, col2 = st.columns(2)

    # ---------------- MISSING VALUES ----------------
    with col1:
        st.markdown("### 🧩 Missing Values Handling")

        option = st.selectbox(
            "Choose Method",
            ["None", "Mean", "Median", "Mode"]
        )

        if option == "Mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("✅ Filled with Mean")

        elif option == "Median":
            df = df.fillna(df.median(numeric_only=True))
            st.success("✅ Filled with Median")

        elif option == "Mode":
            df = df.fillna(df.mode().iloc[0])
            st.success("✅ Filled with Mode")

        else:
            st.info("No missing value handling")

    # ---------------- OUTLIER ----------------
    with col2:
        st.markdown("### 🚨 Outlier Handling")

        outlier = st.checkbox("Remove Outliers (IQR Method)")

        if outlier:

            num_df = df.select_dtypes(include='number')

            if num_df.shape[1] == 0:
                st.warning("⚠️ No numeric columns for outlier removal")
            else:
                Q1 = num_df.quantile(0.25)
                Q3 = num_df.quantile(0.75)
                IQR = Q3 - Q1

                mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)

                df = df[mask]

                st.success("✅ Outliers Removed Successfully")

    # ✅ SAVE BACK (VERY IMPORTANT)
    st.session_state.data = df

    # ---------------- SUCCESS UI ----------------
    st.markdown("""
    <div style='padding:10px; border-radius:10px; background-color:#d4edda'>
        <b>🎉 Cleaning Done Successfully</b>
    </div>
    """, unsafe_allow_html=True)

    st.write("📊 Shape after cleaning:", df.shape)
# =========================
# STEP 5 FEATURE SELECTION
# =========================
elif step == steps[4]:

    st.markdown("""
    <div style='padding:15px; border-radius:10px; background-color:#eef6ff'>
        <h3>🔍 Feature Selection</h3>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.data
    target = st.session_state.target

    if target not in df.columns:
        st.error("Target column missing ❌")
        st.stop()

    X = df.drop(columns=[target, "StudentID"], errors='ignore')
    y = df[target]

    # Encoding
    X = pd.get_dummies(X)
    X = X.fillna(0)
    problem = st.session_state.problem

    if problem == "Regression":
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(y.mean())
    else:
        problem = st.session_state.problem

        if problem == "Regression":
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.mean())
        else:
            y = y.astype(str)  # ✅ prevent weird encoding issues
            y = y.fillna("Unknown").astype("category").cat.codes

    col1, col2 = st.columns(2)

    # ---------------- SELECT METHOD ----------------
    with col1:
        method = st.selectbox(
            "⚙️ Choose Method",
            ["None", "Variance Threshold", "ANOVA", "Z-Score"]
        )

    # ---------------- APPLY ----------------
    with col2:
        st.markdown("### 🚀 Apply Feature Selection")

        if method == "None":
            st.info("No Feature Selection Applied")

        elif method == "Variance Threshold":
            #from sklearn.feature_selection import VarianceThreshold
        
            try:
                selector = VarianceThreshold(0.01)
                X_new = selector.fit_transform(X)
                
                if X_new.shape[1] == 0:
                    st.warning("⚠️ All features removed due to low variance")
                else:
                    X = X_new
                    st.success("✅ Variance Applied")
        
            except Exception as e:
                st.error(f"❌ Variance Threshold Error: {e}")

        elif method == "ANOVA":
            from sklearn.feature_selection import SelectKBest, f_classif
            X = SelectKBest(f_classif, k="all").fit_transform(X, y)

            st.success("✅ ANOVA Applied")

        elif method == "Z-Score":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            st.success("✅ Z-Score Applied")

    # Save
    if X is None or len(X) == 0:
        st.error("❌ Feature selection resulted in empty dataset")
        st.stop()

        # =========================
    # 🤖 AUTO DATA FIX ENGINE
    # =========================
    
    problem = st.session_state.problem
    
    # Convert to pandas for safety
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    # -------- CLASSIFICATION FIX --------
    if problem != "Regression":
    
        class_counts = y.value_counts()
        min_class = class_counts.min()
    
        # ❌ Case 1: Only one class
        if len(class_counts) < 2:
            st.error("❌ Dataset has only one class. Cannot train model.")
            st.stop()
    
        # ⚠️ Case 2: Rare classes (<=1 sample)
        rare_classes = class_counts[class_counts <= 1].index
    
        if len(rare_classes) > 0:
            st.warning(f"⚠️ Removing {len(rare_classes)} rare class(es) with ≤1 sample")
    
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]
    
        # ⚠️ Case 3: Too many unique classes (looks like regression)
        if len(class_counts) > 0.5 * len(y):
            st.warning("⚠️ Too many unique target values → switching to Regression")
    
            st.session_state.problem = "Regression"
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.mean())
    
    # -------- REGRESSION FIX --------
    else:
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(y.mean())
    
    # Final safety
    if len(X) == 0:
        st.error("❌ No data left after cleaning")
        st.stop()
    st.session_state.X = X
    st.session_state.y = y

    st.markdown("""
    <div style='padding:10px; border-radius:10px; background-color:#d1ecf1'>
        <b>✨ Feature Selection Completed</b>
    </div>
    """, unsafe_allow_html=True)

    st.write("📊 Final Shape:", X.shape)
# =========================
# STEP 6 SPLIT
# =========================
elif step == steps[5]:

    X = st.session_state.X
    y = st.session_state.y
    st.markdown('<div class="card">✂️ Train-Test Split</div>', unsafe_allow_html=True)

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    if X is None or y is None:
        st.error("❌ Data not ready. Complete previous steps.")
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if len(np.unique(y_train)) < 2:
        st.error("❌ After split: Only one class in training data")
        st.stop()

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split done ✅")

# =========================
# STEP 7 MODEL
# =========================
elif step == steps[6]:

    problem = st.session_state.problem

    if problem == "Clustering":
        model_name = st.selectbox("Model", ["KMeans", "DBSCAN"])
    elif problem == "Regression":
        model_name = st.selectbox("Model", ["Linear"])
    else:
        model_name = st.selectbox("Model", ["Logistic", "SVM", "RandomForest"])

    st.session_state.model_name = model_name

# =========================
# STEP 8 TRAINING
# =========================
elif step == steps[7]:

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    model_name = st.session_state.model_name

    if len(X_train) == 0:
        st.error("❌ Training data is empty")
        st.stop()

    # Model create
    if model_name == "Logistic":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "SVM":
        model = SVC()

    elif model_name == "RandomForest":
        model = RandomForestClassifier()

    elif model_name == "Linear":
        model = LinearRegression()

    elif model_name == "KMeans":
        model = KMeans(n_clusters=3)

    elif model_name == "DBSCAN":
        model = DBSCAN()

    k = st.slider("K-Fold", 2, 5, 3)

    if st.button("🚀 Train Model"):

        if model_name in ["KMeans", "DBSCAN"]:
            model.fit(X_train)
            st.success("Clustering done 🎉")
            st.balloons()

        else:
            # ✅ CHECK TARGET VALIDITY BEFORE TRAINING
            unique_classes = np.unique(y_train)
            
            if model_name in ["Logistic", "SVM", "RandomForest"]:
                
                if len(unique_classes) < 2:
                    st.error("❌ Training failed: Only 1 class present in data")
                    st.stop()
            
                if len(unique_classes) > 0.5 * len(y_train):
                    st.warning("⚠️ Too many unique values in target → may be regression problem")
            
            model.fit(X_train, y_train)

            from sklearn.model_selection import KFold

            try:
                if model_name == "Linear":
                    cv = KFold(n_splits=k)
            
                else:
                    class_counts = pd.Series(y_train).value_counts()
                    min_class_samples = class_counts.min()
            
                    # ❗ CRITICAL FIX
                    if min_class_samples < 2:
                        st.warning("⚠️ Some classes have only 1 sample → Using KFold instead of StratifiedKFold")
                        cv = KFold(n_splits=2)
            
                    else:
                        if k > min_class_samples:
                            st.warning(f"⚠️ Reduced folds to {min_class_samples}")
                            k = min_class_samples
            
                        cv = StratifiedKFold(n_splits=k)
            
                scores = cross_val_score(model, X_train, y_train, cv=cv)
            
                st.success(f"Accuracy: {scores.mean():.4f}")
            
            except Exception as e:
                st.error(f"❌ Cross Validation Error: {e}")

           # st.success(f"Accuracy: {scores.mean():.4f}")
            st.balloons()

        st.session_state.model = model

# =========================
# STEP 9 METRICS
# =========================
# =========================
# STEP 9 METRICS (FINAL)
# =========================
# =========================
# STEP 9 → METRICS + REPORT
# =========================
elif step == steps[8]:

    # ✅ Safety check
    if "model" not in st.session_state:
        st.warning("⚠️ Please train model first")
        st.stop()

    model = st.session_state.model
    model_name = st.session_state.model_name

    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if st.button("📊 Evaluate Model"):

        # -----------------------
        # CLUSTERING
        # -----------------------
        if model_name in ["KMeans", "DBSCAN"]:
            preds = model.fit_predict(X_test)

            st.success("Clustering Done 🎉")
            st.balloons()

            st.write("Cluster Labels:", preds[:10])

        # -----------------------
        # REGRESSION
        # -----------------------
        elif model_name == "Linear":

            preds = model.predict(X_test)

            from sklearn.metrics import mean_squared_error, r2_score

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            st.success("Regression Metrics 🎯")
            st.balloons()

            metrics_df = pd.DataFrame({
                "Metric": ["MSE", "R2 Score"],
                "Value": [mse, r2]
            })

            st.table(metrics_df)

            # 📥 Download
            csv = metrics_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Report",
                data=csv,
                file_name="regression_report.csv",
                mime="text/csv"
            )

        # -----------------------
        # CLASSIFICATION
        # -----------------------
        else:

            preds = model.predict(X_test)

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix
            )

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            st.success("Classification Metrics 🎯")
            st.balloons()

            # 🔥 METRIC BOX UI
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")

            # 📊 TABLE
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Value": [acc, prec, rec, f1]
            })

            st.subheader("📊 Metrics Table")
            st.table(metrics_df)

            # 📉 CONFUSION MATRIX
            st.subheader("📉 Confusion Matrix")

            cm = confusion_matrix(y_test, preds)

            import seaborn as sns
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

            st.pyplot(fig)

            # 📥 DOWNLOAD REPORT
            csv = metrics_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Report",
                data=csv,
                file_name="classification_report.csv",
                mime="text/csv"
            )
