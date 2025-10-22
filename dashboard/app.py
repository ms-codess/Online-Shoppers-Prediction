import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "online_shoppers_intention.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "Revenue") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Known categoricals in the UCI dataset
    categorical = [
        col
        for col in ["Month", "VisitorType", "Weekend"]
        if col in X.columns
    ]
    # Everything else numeric by default
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_pipeline(model_name: str, numeric: List[str], categorical: List[str]) -> Pipeline:
    # Match notebook-style preprocessing: impute + label-like encoding + scaling
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, n_jobs=None)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("Unsupported model")

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("scale", StandardScaler()),  # scale all features after encoding, like notebook
        ("model", model),
    ])
    return pipe


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["roc_auc"] = np.nan
    return out


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)


def get_feature_names(pipe: Pipeline) -> Optional[List[str]]:
    try:
        prep = pipe.named_steps["prep"]
        return list(prep.get_feature_names_out())
    except Exception:
        return None


def show_feature_importance(pipe: Pipeline):
    model = pipe.named_steps["model"]
    names = get_feature_names(pipe)
    values = None
    title = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        title = "Feature Importance (Gini)"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)[0]
        title = "Feature Importance (|coeff|)"

    if values is None:
        st.info("Model does not expose feature importances.")
        return

    if names is None or len(names) != len(values):
        names = [f"f{i}" for i in range(len(values))]

    imp = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    top_k = st.slider("Show top features", 5, min(30, len(imp)), 15)
    st.dataframe(imp.head(top_k), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, max(3, int(top_k * 0.3))))
    sns.barplot(
        data=imp.head(top_k),
        x="importance",
        y="feature",
        palette="viridis",
        ax=ax,
    )
    ax.set_title(title)
    st.pyplot(fig)


def sidebar_config(df: pd.DataFrame) -> dict:
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Model",
        ["Logistic Regression", "Random Forest"],
        index=1,
    )
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)
    return {
        "model_name": model_name,
        "test_size": test_size,
        "random_state": int(random_state),
    }


def nb_style_clean_preview(df: pd.DataFrame) -> pd.DataFrame:
    # Mirror basic cleaning steps performed in the notebook for preview: drop duplicates
    # Imputation and encoding are handled inside the model pipeline for parity
    return df.drop_duplicates()


def plot_class_balance(y: pd.Series):
    vals = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=vals.index.astype(str), y=vals.values, ax=ax, palette="pastel")
    ax.set_xlabel("Revenue (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance")
    st.pyplot(fig)


def plot_corr_heatmap(df: pd.DataFrame):
    # Encode object columns temporarily for correlation calculation
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == "object":
            df_enc[col] = pd.factorize(df_enc[col])[0]
        elif str(df_enc[col].dtype) == "bool":
            df_enc[col] = df_enc[col].astype(int)
    corr = df_enc.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation (Notebook-style)")
    st.pyplot(fig)


def render_single_predict_form(X: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Single Prediction")
    cols = st.columns(2)
    inputs = {}

    for i, col in enumerate(X.columns):
        c = cols[i % 2]
        if X[col].dtype.kind in "iuf":
            default = float(np.nan_to_num(X[col].median()))
            inputs[col] = c.number_input(col, value=default)
        else:
            choices = sorted([str(v) for v in X[col].dropna().unique().tolist()])
            default_idx = 0
            if len(choices) > 1 and "FALSE" in choices:
                default_idx = choices.index("FALSE")
            inputs[col] = c.selectbox(col, options=choices, index=default_idx)

    predict_btn = st.button("Predict")
    if predict_btn:
        return pd.DataFrame([inputs])
    return pd.DataFrame()


def main():
    st.set_page_config(page_title="Online Shoppers Dashboard", layout="wide")
    st.title("Online Shoppers Intention — Interactive Dashboard")
    st.caption("Predict likelihood to buy; train, evaluate, and visualize performance.")

    default_path = os.path.normpath(DATA_PATH_DEFAULT)
    st.sidebar.write(f"Data: `{os.path.relpath(default_path, os.getcwd())}`")

    # Load data
    df_default = load_data(default_path)
    config = sidebar_config(df_default)

    # Always train/evaluate using raw-like schema; apply preprocessing in-pipeline.
    # If a cleaned CSV exists, we note it but still use the raw CSV for consistent UX.
    cleaned_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "online_shoppers_intention_cleaned.csv"))
    if os.path.exists(cleaned_path):
        st.sidebar.info("Detected cleaned CSV (for sharing/repro). Dashboard applies same steps internally for live predictions.")
    df = df_default.copy()

    # Basic checks and setup
    has_target = "Revenue" in df.columns
    if not has_target:
        st.warning("Column 'Revenue' not found. Add it for evaluation; predictions still work.")

    # Prepare features/target
    if has_target:
        X, y = split_features_target(df)
        # Coerce y to 0/1
        y = y.map(lambda v: 1 if str(v).strip().upper() in {"TRUE", "1", "YES"} else 0)
    else:
        X = df.copy()
        y = None

    numeric, categorical = infer_column_types(X)

    # Train/test split if we can evaluate
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, None, None, None

    # Build and train pipeline
    pipe = build_pipeline(config["model_name"], numeric, categorical)

    with st.spinner("Training model..."):
        if y is not None:
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X, np.zeros(len(X)))

    st.success("Model ready.")

    # UI Tabs: Overview, Predict, Train & Evaluate, Compare Models, Notebook Results, Details
    tab_overview, tab_predict, tab_eval, tab_compare, tab_nb, tab_details = st.tabs([
        "Overview", "Predict", "Train & Evaluate", "Compare Models", "Notebook Results", "Details"
    ])

    with tab_overview:
        st.subheader("Problem & Solution")
        st.markdown(
            """
            - Problem: Predict whether a shopping session will end in a purchase.
            - Solution: Train ML models on session features to estimate BUY probability, then classify using a decision threshold.
            - Audience: Product, Growth, and CRO teams seeking better conversion insights.
            - What to do here: Use Predict for one-off scoring; use Compare to pick a model; use Train & Evaluate to inspect performance.
            """
        )

    with tab_data:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(
            f"Rows: {len(X)} | Numeric: {len(numeric)} | Categorical: {len(categorical)}"
        )
        if st.checkbox("Show cleaned preview (drop duplicates)"):
            dfc = nb_style_clean_preview(df)
            st.write(f"Rows after de-duplication: {len(dfc)}")
            st.dataframe(dfc.head(20), use_container_width=True)

    with tab_nb:
        st.subheader("Notebook Results — Key Insights")
        st.caption("Replicates core visuals from the notebook with a brief explanation.")
        if y is not None:
            plot_class_balance(y)
            st.markdown("Class balance helps gauge if the dataset is imbalanced, which affects metric choice and threshold.")

        st.markdown("---")
        st.subheader("Correlation Heatmap")
        plot_corr_heatmap(df)
        st.markdown("Correlation highlights linear relationships; strong correlations can inform feature selection and model choice.")

        st.markdown("---")
        st.subheader("Distributions of Key Features")
        feat = st.selectbox(
            "Choose a feature to view distribution",
            [
                c for c in [
                    "ProductRelated_Duration",
                    "ProductRelated",
                    "PageValues",
                    "BounceRates",
                    "ExitRates",
                    "SpecialDay",
                ] if c in df.columns
            ]
        )
        if feat:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df[feat], kde=True, ax=ax, color="#4c78a8")
            ax.set_title(f"Distribution: {feat}")
            st.pyplot(fig)
            st.markdown("Distributions reveal skew/outliers; consider scaling or robust models accordingly.")

    with tab_eval:
        st.subheader("Evaluation")
        if y is not None:
            y_pred = pipe.predict(X_test)
            y_prob = None
            try:
                y_prob = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

            metrics = compute_metrics(y_test, y_pred, y_prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['precision']:.3f}")
            c3.metric("Recall", f"{metrics['recall']:.3f}")
            c1.metric("F1", f"{metrics['f1']:.3f}")
            if not np.isnan(metrics.get("roc_auc", np.nan)):
                c2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

            st.subheader("Confusion Matrix")
            plot_confusion(y_test.to_numpy(), y_pred)

            if y_prob is not None:
                col_roc, col_pr = st.columns(2)
                with col_roc:
                    st.subheader("ROC Curve")
                    plot_roc(y_test.to_numpy(), y_prob)
                with col_pr:
                    st.subheader("Precision-Recall Curve")
                    plot_pr(y_test.to_numpy(), y_prob)

            st.subheader("Feature Importance")
            show_feature_importance(pipe)
        else:
            st.info("No target column detected; evaluation metrics are hidden.")

    with tab_compare:
        st.subheader("Model Comparison")
        st.caption("Trains Logistic Regression and Random Forest with the same preprocessing and split, then compares metrics.")
        if y is not None:
            # Build two pipelines
            pipe_lr = build_pipeline("Logistic Regression", numeric, categorical)
            pipe_rf = build_pipeline("Random Forest", numeric, categorical)
            with st.spinner("Training models for comparison..."):
                pipe_lr.fit(X_train, y_train)
                pipe_rf.fit(X_train, y_train)

            def eval_model(p):
                yp = p.predict(X_test)
                try:
                    pr = p.predict_proba(X_test)[:, 1]
                except Exception:
                    pr = None
                return compute_metrics(y_test, yp, pr)

            m_lr = eval_model(pipe_lr)
            m_rf = eval_model(pipe_rf)
            comp = pd.DataFrame([m_lr, m_rf], index=["Logistic Regression", "Random Forest"]).round(3)
            st.dataframe(comp, use_container_width=True)

            metric_to_plot = st.selectbox("Metric to chart", [c for c in comp.columns if c in ["roc_auc", "f1", "accuracy", "precision", "recall"]], index=0 if "roc_auc" in comp.columns else 0)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=comp.index, y=comp[metric_to_plot], ax=ax, palette="viridis")
            ax.set_ylabel(metric_to_plot)
            ax.set_xlabel("Model")
            ax.set_title("Side-by-side comparison")
            st.pyplot(fig)
        else:
            st.info("Metrics unavailable without the 'Revenue' target column.")

    with tab_predict:
        st.subheader("Predict — Will the shopper buy?")
        st.caption("Set session details below, adjust decision threshold, and click Predict.")

        months_from_data = None
        if "Month" in X.columns:
            months_from_data = [m for m in pd.Series(X["Month"].dropna().unique()).astype(str).tolist()]
        default_months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months = months_from_data or default_months
        visit_types = ["Returning_Visitor", "New_Visitor", "Other"]
        tf = ["FALSE", "TRUE"]

        with st.form(key="predict_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                admin = st.number_input(
                    "Admin pages visited",
                    min_value=0,
                    value=int(X.get("Administrative", pd.Series([0])).median() or 0),
                    help="Number of account/help pages viewed."
                )
                info = st.number_input(
                    "Info pages visited",
                    min_value=0,
                    value=int(X.get("Informational", pd.Series([0])).median() or 0),
                    help="Number of information pages viewed (policies, FAQ, etc.)."
                )
                prod = st.number_input(
                    "Product pages visited",
                    min_value=0,
                    value=int(X.get("ProductRelated", pd.Series([1])).median() or 1),
                    help="How many product pages were viewed."
                )

            with col2:
                admin_d = st.number_input(
                    "Time on admin pages (sec)",
                    min_value=0.0,
                    value=float(X.get("Administrative_Duration", pd.Series([0.0])).median() or 0.0),
                    help="Total time spent on account/help pages."
                )
                info_d = st.number_input(
                    "Time on info pages (sec)",
                    min_value=0.0,
                    value=float(X.get("Informational_Duration", pd.Series([0.0])).median() or 0.0),
                    help="Total time spent on information pages."
                )
                prod_d = st.number_input(
                    "Time on product pages (sec)",
                    min_value=0.0,
                    value=float(X.get("ProductRelated_Duration", pd.Series([0.0])).median() or 0.0),
                    help="Total time spent viewing products."
                )

            with col3:
                bounce = st.number_input(
                    "Immediate exits (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(min(1.0, max(0.0, (X.get("BounceRates", pd.Series([0.05])).median() or 0.05)))),
                    help="Higher means more sessions ended right away."
                )
                exit_r = st.number_input(
                    "Exit rate (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(min(1.0, max(0.0, (X.get("ExitRates", pd.Series([0.1])).median() or 0.1)))),
                    help="Fraction of page views that ended the session."
                )
                page_val = st.number_input(
                    "Average page value",
                    min_value=0.0,
                    value=float(X.get("PageValues", pd.Series([0.0])).median() or 0.0),
                    help="Higher when pages are near conversion steps."
                )

            col4, col5, col6, col7 = st.columns(4)
            with col4:
                special = st.number_input(
                    "Near special day (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(min(1.0, max(0.0, (X.get("SpecialDay", pd.Series([0.0])).median() or 0.0)))),
                    help="Closer to 1 if session is near a special day (e.g., holidays)."
                )
                month = st.selectbox(
                    "Month of visit",
                    options=months,
                    index=0 if "Month" not in X.columns else 0,
                    help="Month when the session occurred."
                )
            with col5:
                os_val = st.number_input(
                    "Operating system (code)",
                    min_value=1,
                    value=int(X.get("OperatingSystems", pd.Series([1])).median() or 1),
                    help="Encoded OS identifier from dataset."
                )
                browser = st.number_input(
                    "Browser (code)",
                    min_value=1,
                    value=int(X.get("Browser", pd.Series([1])).median() or 1),
                    help="Encoded browser identifier."
                )
            with col6:
                region = st.number_input(
                    "Region (code)",
                    min_value=1,
                    value=int(X.get("Region", pd.Series([1])).median() or 1),
                    help="Encoded region identifier."
                )
                traffic = st.number_input(
                    "Traffic source (code)",
                    min_value=1,
                    value=int(X.get("TrafficType", pd.Series([1])).median() or 1),
                    help="Encoded traffic source identifier."
                )
            with col7:
                vtype = st.selectbox(
                    "Visitor type",
                    options=visit_types,
                    index=0,
                    help="Is this a returning or new visitor?"
                )
                weekend = st.selectbox(
                    "Weekend session?",
                    options=tf,
                    index=0,
                    help="Whether the session was on a weekend."
                )

            threshold = st.slider(
                "Decision threshold",
                0.0, 1.0, 0.5, 0.01,
                help="Classify as BUY when predicted probability is at or above this value."
            )

            submit = st.form_submit_button("Predict")

        if submit:
            row = {
                "Administrative": admin,
                "Administrative_Duration": admin_d,
                "Informational": info,
                "Informational_Duration": info_d,
                "ProductRelated": prod,
                "ProductRelated_Duration": prod_d,
                "BounceRates": bounce,
                "ExitRates": exit_r,
                "PageValues": page_val,
                "SpecialDay": special,
                "Month": month,
                "OperatingSystems": int(os_val),
                "Browser": int(browser),
                "Region": int(region),
                "TrafficType": int(traffic),
                "VisitorType": vtype,
                "Weekend": weekend,
            }
            single = pd.DataFrame([row])
            try:
                prob = float(pipe.predict_proba(single)[:, 1][0])
            except Exception:
                prob = None
            pred = int(pipe.predict(single)[0])
            label = (1 if (prob is not None and prob >= threshold) else pred) if prob is not None else pred

            c_left, c_right = st.columns([2, 1])
            with c_left:
                if label == 1:
                    st.success("Prediction: BUY")
                else:
                    st.warning("Prediction: NOT BUY")
                if prob is not None:
                    st.write(f"Predicted probability of BUY: {prob:.3f}")
                    st.progress(min(1.0, max(0.0, prob)))
            with c_right:
                st.metric("Threshold", f"{threshold:.2f}")
                if prob is not None:
                    st.metric("Decision", "BUY" if label == 1 else "NOT BUY")

            st.caption("Tip: increase the threshold for fewer false positives; lower it for fewer missed buyers.")

    with tab_details:
        st.subheader("How It Works")
        st.markdown(
            """
            - Data source: the UCI Online Shoppers dataset (CSV included in this project).
            - Target: `Revenue` — whether the session led to a purchase.
            - Cleaning (as in the notebook):
              - Drop duplicate rows.
              - Impute missing values (categorical → most frequent; numeric → median).
            - Feature engineering (notebook-style):
              - Categorical features (`Month`, `VisitorType`, `Weekend`) are label/ordinal encoded.
              - All features are scaled (StandardScaler) before modeling.
              - These steps are implemented inside the model pipeline so live predictions use the same transformations as training.
            - Cleaned CSV:
              - You may generate `online_shoppers_intention_cleaned.csv` via `tools/prepare_cleaned_csv.py` for sharing or offline analysis.
              - The dashboard still performs the same steps internally for consistency and simplicity.
            - Models:
              - Logistic Regression (interpretable coefficients)
              - Random Forest (nonlinear, exposes feature importances)
            - Training/Evaluation:
              - The dataset is split into Train/Test using the slider in the sidebar (stratified by target).
              - Metrics shown: Accuracy, Precision, Recall, F1, ROC AUC; plus Confusion Matrix, ROC and PR curves.
            - Threshold:
              - In Predict, you set the decision threshold. A session is classified as BUY when the predicted probability ≥ threshold.
            - Feature meanings (simplified):
              - Admin/Info/Product pages visited & time: how much the shopper explored different page types.
              - Immediate exits (Bounce) and Exit rate: higher values typically reduce likelihood to buy.
              - Average page value: higher near conversion steps.
              - Special day: proximity to a special day (e.g., holiday).
              - Month/OS/Browser/Region/Traffic: technical/context codes captured in the dataset.
            """
        )


if __name__ == "__main__":
    main()
