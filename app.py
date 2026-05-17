import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease – Stacking Classifier",
    page_icon="❤️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (improved dark-mode readability for labels and cards)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
/* Nav radio labels */
[data-testid="stSidebar"] label {
    font-size: 15px !important;
    font-weight: 500 !important;
}
/* Metric cards */
[data-testid="metric-container"] {
    background-color: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Section headers */
h2, h3 { color: #1a1a2e; }

/* Feature card (dark-mode friendly) */
.feat-card {
    background: #1a1a2e;
    border-left: 4px solid #4361ee;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #e8e8f0 !important;
}
.feat-card b { color: #FFD700 !important; }
.feat-card span { color: #e8e8f0 !important; }

/* Slider label fix for dark mode */
div[data-testid="stSlider"] label {
    color: #e8e8f0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Selectbox label fix for dark mode */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] .css-1aumxhk {
    color: #e8e8f0 !important;
    font-weight: 600 !important;
}

/* Prediction result box */
.pred-box {
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
    font-size: 18px;
    font-weight: 600;
}
.pred-disease  { background: #ffe0e0; border-left: 6px solid #e63946; color: #7b0000; }
.pred-healthy  { background: #d4edda; border-left: 6px solid #2a9d8f; color: #155724; }

/* Divider */
hr { border: none; border-top: 1px solid #e0e0e0; margin: 18px 0; }

/* Ensure metric text is readable */
[data-testid="metric-container"] .stMetricValue {
    color: #0b1220 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Heart_Disease_Dataset.csv")

    # ── Missing value fill ────────────────────────────────────────────────────
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # ── Outlier handling (IQR clip) ───────────────────────────────────────────
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df_before = df.copy()
    for col in continuous_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    df_after = df.copy()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Build stacking ────────────────────────────────────────────────────────
    stacking = StackingClassifier(
        estimators=[
            ("KNN",           KNeighborsClassifier(n_neighbors=5)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("SVM",           SVC(probability=True, random_state=42)),
            ("Naive Bayes",   GaussianNB()),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
    )

    base_models = {
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "SVM":           SVC(probability=True, random_state=42),
        "Naive Bayes":   GaussianNB(),
        "Stacking":      stacking,
    }

    results = {}
    for name, mdl in base_models.items():
        mdl.fit(X_train_sc, y_train)
        pred = mdl.predict(X_test_sc)
        results[name] = {
            "model":  mdl,
            "pred":   pred,
            "acc":    accuracy_score(y_test, pred),
            "cm":     confusion_matrix(y_test, pred),
            "report": classification_report(y_test, pred, output_dict=True),
        }

    return (df, df_before, df_after, X, y,
            X_train_sc, X_test_sc, y_train, y_test,
            scaler, results, continuous_cols)

(df, df_before, df_after, X, y,
 X_train_sc, X_test_sc, y_train, y_test,
 scaler, results, continuous_cols) = load_and_train()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (display_name, description, widget, *widget_args, default)
FEATURE_INFO = {
    "age":      ("Age (years)",              "Patient age in years. Older age increases heart disease risk.",
                 "slider_int", 29, 77, int(df["age"].mean())),
    "sex":      ("Sex",                      "1 = Male, 0 = Female. Males have higher risk.",
                 "select", [0, 1], ["Female (0)", "Male (1)"], 1),
    "cp":       ("Chest Pain Type",          "0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic.",
                 "select", [0,1,2,3], ["Typical Angina (0)","Atypical Angina (1)","Non-anginal Pain (2)","Asymptomatic (3)"], 0),
    "trestbps": ("Resting Blood Pressure",   "Resting BP in mm Hg. Normal: <120. High BP strains the heart.",
                 "slider_int", 94, 200, int(df["trestbps"].mean())),
    "chol":     ("Serum Cholesterol (mg/dl)","Cholesterol level. Normal: <200. High cholesterol blocks arteries.",
                 "slider_int", 126, 564, int(df["chol"].mean())),
    "fbs":      ("Fasting Blood Sugar",      "Blood sugar >120 mg/dl fasting? 1=Yes (diabetic risk), 0=No.",
                 "select", [0, 1], ["No (0)", "Yes (1)"], 0),
    "restecg":  ("Resting ECG Result",       "0=Normal, 1=ST-T Wave Abnormality, 2=Left Ventricular Hypertrophy.",
                 "select", [0,1,2], ["Normal (0)","ST-T Abnormality (1)","LV Hypertrophy (2)"], 0),
    "thalach":  ("Max Heart Rate Achieved",  "Peak heart rate during exercise test. Lower values may indicate cardiac issues.",
                 "slider_int", 71, 202, int(df["thalach"].mean())),
    "exang":    ("Exercise Induced Angina",  "Chest pain during exercise? 1=Yes (concerning sign), 0=No.",
                 "select", [0, 1], ["No (0)", "Yes (1)"], 0),
    "oldpeak":  ("ST Depression",            "ST depression caused by exercise vs rest. Higher value = more concern.",
                 "slider_float", 0.0, 6.2, round(float(df["oldpeak"].mean()), 1)),
    "slope":    ("Slope of ST Segment",      "0=Upsloping (better), 1=Flat, 2=Downsloping (more concern).",
                 "select", [0,1,2], ["Upsloping (0)","Flat (1)","Downsloping (2)"], 1),
    "ca":       ("Major Vessels Coloured",   "Number of major blood vessels coloured by fluoroscopy (0–4). More = worse.",
                 "select", [0,1,2,3,4], ["0","1","2","3","4"], 0),
    "thal":     ("Thalassemia",              "Blood disorder type. 0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown.",
                 "select", [0,1,2,3], ["Normal (0)","Fixed Defect (1)","Reversible Defect (2)","Unknown (3)"], 2),
}

features = [f for f in df.columns if f != "target"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Disease")
    st.markdown("### Stacking Classifier")
    st.markdown("---")

    # Navigation at TOP of sidebar
    page = st.radio(
        "📌 Go to",
        ["📖 What is Stacking?", "📊 Stacking Classifier", "🔮 Predict"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"🗂 Rows &nbsp;&nbsp;&nbsp;: **{df.shape[0]}**")
    st.markdown(f"📋 Columns : **{df.shape[1]}**")
    st.markdown(f"🎯 Target &nbsp;: Heart Disease (0/1)")
    st.markdown("---")
    st.markdown("**Model Accuracies**")
    for name, res in results.items():
        st.markdown(f"• {name}: **{res['acc']*100:.1f}%**")
    st.markdown("---")
    st.caption("Built with Streamlit · Sklearn")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – WHAT IS STACKING?
# ─────────────────────────────────────────────────────────────────────────────
if page == "📖 What is Stacking?":
    st.title("📖 What is Stacking?")
    st.markdown("---")

    st.subheader("Simple Explanation")
    st.info(
        "**Stacking** (Stacked Generalisation) is an ensemble learning method. "
        "It trains several different models — called **base learners** — on the same data. "
        "Then, a separate model called the **meta learner** learns how to best combine "
        "their predictions into a final, more accurate answer.\n\n"
        "Think of it like asking 4 different doctors for their opinion, then having an "
        "expert consultant make the final diagnosis based on all 4 opinions."
    )

    st.markdown("---")
    st.subheader("How Stacking Works — Step by Step")
    st.markdown("""
1. **Split** the training data using cross-validation (e.g. 5 folds).
2. **Train each base learner** (KNN, Decision Tree, SVM, Naive Bayes) on the training folds.
3. **Collect out-of-fold predictions** from each base learner.
4. **Stack predictions** into a new feature matrix — each column = one base learner's output.
5. **Train the meta learner** (Logistic Regression) on this stacked feature matrix.
6. **Predict**: base learners predict → meta learner combines → final output.
    """)

    st.markdown("---")
    st.subheader("Base Learners & Meta Learner Used")

    c1, c2 = st.columns([2, 1])
    with c1:
        learners_df = pd.DataFrame({
            "Role":  ["Base Learner", "Base Learner", "Base Learner", "Base Learner", "Meta Learner"],
            "Model": ["KNN (K-Nearest Neighbours)", "Decision Tree", "SVM (Support Vector Machine)", "Naive Bayes", "Logistic Regression"],
            "What it does": [
                "Predicts based on the K closest training samples.",
                "Makes a tree of yes/no decisions to classify.",
                "Finds the best separating boundary between classes.",
                "Uses Bayes' theorem assuming features are independent.",
                "Learns optimal weights to combine base learner outputs.",
            ],
        })
        st.table(learners_df)
    with c2:
        st.markdown("**Formula**")
        st.latex(r"\hat{y} = f_{\text{meta}}\!\left(h_1(x), h_2(x), \ldots, h_k(x)\right)")
        st.caption("hₖ = k-th base learner  |  f_meta = meta learner")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**✅ Advantages**")
        st.markdown("- Combines strengths of multiple models\n- Usually outperforms any single model\n- Works with any combination of algorithms")
    with col2:
        st.markdown("**⚠️ Limitations**")
        st.markdown("- Slower training time\n- Harder to interpret\n- Risk of overfitting on very small data")
    with col3:
        st.markdown("**🌍 Real-world Uses**")
        st.markdown("- Medical diagnosis\n- Fraud detection\n- Kaggle ML competitions\n- Credit risk scoring")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – STACKING CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Stacking Classifier":
    st.title("📊 Stacking Classifier – Heart Disease Dataset")
    st.markdown("---")

    # ── Dataset Overview ──────────────────────────────────────────────────────
    st.subheader("1. Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",      df.shape[0])
    c2.metric("Total Columns",   df.shape[1])
    c3.metric("No Disease (0)",  int((y == 0).sum()))
    c4.metric("Heart Disease (1)", int((y == 1).sum()))

    st.markdown("**First 5 rows of the dataset**")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("**Column Descriptions**")
    col_desc_df = pd.DataFrame(
        [(feat, FEATURE_INFO[feat][0], FEATURE_INFO[feat][1]) for feat in features],
        columns=["Column", "Name", "Description"],
    )
    st.dataframe(col_desc_df, use_container_width=True, hide_index=True)

    with st.expander("📋 Basic Statistics (describe)"):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # ── Target Distribution ───────────────────────────────────────────────────
    st.subheader("2. Target Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.set_style("whitegrid")
    target_counts = df["target"].value_counts()
    bars = ax.bar(["No Disease (0)", "Heart Disease (1)"],
                  target_counts.values,
                  color=["#4878d0", "#ee854a"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Target Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(target_counts.values) + 60)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Missing Values ────────────────────────────────────────────────────────
    st.subheader("3. Missing Values")
    missing = df_before.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in this dataset.")
    else:
        st.dataframe(
            missing[missing > 0].reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
            use_container_width=True,
        )
        st.info("Missing values filled with column median (numerical) or mode (categorical).")

    st.markdown("---")

    # ── Outlier Handling ──────────────────────────────────────────────────────
    st.subheader("4. Outlier Handling — Before vs After (IQR Clipping)")
    st.markdown(
        "Outliers are clipped using the IQR rule: values outside "
        "`[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` are capped. "
        "Box plots show each continuous column **before** and **after** clipping."
    )

    for col in continuous_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        sns.set_style("whitegrid")

        # Before
        axes[0].boxplot(df_before[col].dropna(),
                        patch_artist=True,
                        boxprops=dict(facecolor="#4878d0", color="#23314d"),
                        medianprops=dict(color="red", linewidth=2),
                        whiskerprops=dict(color="#23314d"),
                        capprops=dict(color="#23314d"),
                        flierprops=dict(marker="o", color="red", alpha=0.5))
        axes[0].set_title(f"{col} — Before Outlier Removal", fontsize=11)
        axes[0].set_ylabel(col)

        # After
        axes[1].boxplot(df_after[col].dropna(),
                        patch_artist=True,
                        boxprops=dict(facecolor="#ee854a", color="#23314d"),
                        medianprops=dict(color="red", linewidth=2),
                        whiskerprops=dict(color="#23314d"),
                        capprops=dict(color="#23314d"),
                        flierprops=dict(marker="o", color="red", alpha=0.5))
        axes[1].set_title(f"{col} — After Outlier Removal", fontsize=11)
        axes[1].set_ylabel(col)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Feature Distributions ─────────────────────────────────────────────────
    st.subheader("5. Feature Distributions by Target")
    st.markdown("Histograms showing how each continuous feature is distributed for each class.")

    palette = {0: "#4878d0", 1: "#ee854a"}
    for col in continuous_cols:
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.set_style("whitegrid")
        for tval, color in palette.items():
            subset = df[df["target"] == tval][col]
            ax.hist(subset, bins=20, alpha=0.6, color=color,
                    label="No Disease" if tval == 0 else "Heart Disease", edgecolor="white")
        ax.set_title(f"Distribution of {col} by Target", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.subheader("6. Model Training & Accuracy Comparison")

    acc_df = pd.DataFrame(
        [(name, round(res["acc"] * 100, 2)) for name, res in results.items()],
        columns=["Model", "Accuracy (%)"],
    ).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.set_style("whitegrid")
    colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#b47cc7"]
    bars = ax.bar(acc_df["Model"], acc_df["Accuracy (%)"],
                  color=colors[:len(acc_df)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, acc_df["Accuracy (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Accuracy Comparison — All Models", fontsize=13)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 100)
    ax.set_xlabel("Model")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Per-model details in tabs ─────────────────────────────────────────────
    st.subheader("7. Per-Model Performance Details")
    tabs = st.tabs(list(results.keys()))

    for tab, (name, res) in zip(tabs, results.items()):
        with tab:
            st.metric(f"{name} — Test Accuracy", f"{res['acc']*100:.2f}%")

            col_a, col_b = st.columns([1, 1])

            # Confusion Matrix
            with col_a:
                st.markdown("**Confusion Matrix**")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.set_style("whitegrid")
                sns.heatmap(
                    res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    ax=ax, linewidths=0.5,
                )
                ax.set_xlabel("Predicted Label", fontsize=10)
                ax.set_ylabel("True Label", fontsize=10)
                ax.set_title(f"{name} – Confusion Matrix", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Classification Report
            with col_b:
                st.markdown("**Classification Report**")
                report_df = pd.DataFrame(res["report"]).transpose().round(2)
                st.dataframe(report_df, use_container_width=True)

            # Actual vs Predicted
            st.markdown("**Actual vs Predicted (first 60 test samples)**")
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            sns.set_style("whitegrid")
            x_idx = np.arange(60)
            ax2.plot(x_idx, y_test[:60].values,
                     label="Actual", marker="o", markersize=4,
                     linewidth=1.2, color="#4878d0")
            ax2.plot(x_idx, res["pred"][:60],
                     label="Predicted", marker="x", markersize=5,
                     linewidth=1.2, linestyle="--", color="#ee854a")
            ax2.set_title(f"{name} — Actual vs Predicted Labels", fontsize=12)
            ax2.set_xlabel("Sample Index")
            ax2.set_ylabel("Class (0 = No Disease, 1 = Disease)")
            ax2.set_yticks([0, 1])
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title("🔮 Heart Disease Prediction")
    st.markdown("---")

    # ── Model selector at the TOP ─────────────────────────────────────────────
    st.subheader("Step 1 — Choose Your Model")
    model_choice = st.selectbox(
        "Select the model to use for prediction",
        options=list(results.keys()),
        index=list(results.keys()).index("Stacking"),
        help="Stacking combines all base models. You can also try KNN, SVM, etc. individually.",
    )

    # Show selected model accuracy
    chosen_acc = results[model_choice]["acc"] * 100
    st.info(f"Selected model: **{model_choice}** | Test Accuracy: **{chosen_acc:.2f}%**")

    st.markdown("---")
    st.subheader("Step 2 — Enter Patient Details")
    st.caption("Every input field includes a brief description so you know exactly what to enter.")

    # ── Input fields in 2 columns ─────────────────────────────────────────────
    user_input = {}
    left, right = st.columns(2)

    for i, feat in enumerate(features):
        info        = FEATURE_INFO[feat]
        disp_name   = info[0]
        description = info[1]
        wtype       = info[2]
        col         = left if i % 2 == 0 else right

        with col:
            st.markdown(
                f'<div class="feat-card"><b>{disp_name}</b><br>'
                f'<span style="color:#e8e8f0;font-size:12px">{description}</span></div>',
                unsafe_allow_html=True,
            )

            if wtype == "slider_int":
                _, _, _, lo, hi, default = info
                # ensure sensible bounds
                min_val = int(lo)
                max_val = int(hi)
                default_val = int(default)
                if min_val == max_val:
                    min_val = max(0, min_val - 1)
                    max_val = max_val + 1
                user_input[feat] = st.slider(
                    label=disp_name,
                    min_value=min_val, max_value=max_val, value=default_val,
                    label_visibility="collapsed",
                )

            elif wtype == "slider_float":
                _, _, _, lo, hi, default = info
                min_val = float(lo)
                max_val = float(hi)
                default_val = float(default)
                if min_val == max_val:
                    min_val = max(0.0, min_val - 0.1)
                    max_val = max_val + 0.1
                user_input[feat] = st.slider(
                    label=disp_name,
                    min_value=min_val, max_value=max_val,
                    value=default_val, step=0.1,
                    label_visibility="collapsed",
                )

            elif wtype == "select":
                _, _, _, values, labels, default_val = info
                # ensure default index exists
                try:
                    default_index = values.index(default_val)
                except ValueError:
                    default_index = 0
                chosen_label = st.selectbox(
                    label=disp_name,
                    options=labels,
                    index=default_index,
                    label_visibility="collapsed",
                )
                user_input[feat] = values[labels.index(chosen_label)]

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    st.subheader("Step 3 — Get Prediction")

    if st.button("🔮 Run Prediction", use_container_width=True, type="primary"):
        input_df  = pd.DataFrame([user_input])[features]
        input_sc  = scaler.transform(input_df)
        mdl       = results[model_choice]["model"]
        pred      = mdl.predict(input_sc)[0]
        # Some models (like SVM without probability) may not support predict_proba
        try:
            proba     = mdl.predict_proba(input_sc)[0]
            conf      = proba[pred] * 100
        except Exception:
            proba = None
            conf = None

        # ── Result display ────────────────────────────────────────────────────
        if pred == 1:
            st.markdown(
                '<div class="pred-box pred-disease">'
                '⚠️ &nbsp; Prediction: <b>HEART DISEASE DETECTED</b>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-box pred-healthy">'
                '✅ &nbsp; Prediction: <b>NO HEART DISEASE</b>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Used",    model_choice)
        m2.metric("Prediction",    "Heart Disease" if pred == 1 else "No Disease")
        if conf is not None:
            m3.metric("Confidence",    f"{conf:.1f}%")
        else:
            m3.metric("Confidence",    "N/A")

        st.markdown("---")

        # Probability bar chart (if available)
        if proba is not None:
            st.markdown("**Class Probability Breakdown**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            sns.set_style("whitegrid")
            bars = ax.barh(
                ["No Disease (0)", "Heart Disease (1)"],
                proba,
                color=["#4878d0", "#ee854a"],
                edgecolor="white",
            )
            for bar, val in zip(bars, proba):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val*100:.1f}%", va="center", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Probability breakdown not available for the selected model.")

        st.markdown("---")

        # Show input summary
        st.subheader("Input Summary")
        input_summary = {feat: user_input[feat] for feat in features}
        st.json(input_summary)

        st.markdown("---")

        # Explain model context
        st.subheader("Model Context")
        st.markdown(
            "The model predicts based on historical patterns in the dataset. "
            "Use the accuracy and classification metrics shown earlier to gauge typical performance. "
            "If you need higher confidence, try the Stacking model or validate on more data."
        )

# End of file
