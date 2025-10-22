import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "online_shoppers_intention.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "Revenue") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [c for c in ["Month", "VisitorType", "Weekend"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_pipeline(model_name: str, numeric: List[str], categorical: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("Unsupported model")

    pipe = Pipeline(steps=[("prep", preprocessor), ("scale", StandardScaler()), ("model", model)])
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


def nb_style_clean_preview(df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    st.set_page_config(page_title="Online Shoppers Dashboard", page_icon="🛍️", layout="wide")
    # Simple, tasteful theming for headings and font
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        }
        h1, h2, h3 { font-weight: 700; }
        .tag {
            display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600;
            background: #E6F4EA; color:#137333; margin-right:6px; border:1px solid #C7E9D3;
        }
        .callout { background:#F8FAFF; border:1px solid #E3E8FF; padding:12px 14px; border-radius:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🛍️ Online Shoppers Intention — Interactive Dashboard")
    st.caption("Predict purchase likelihood; train, evaluate, compare models, and explore insights.")

    # Load data
    default_path = os.path.normpath(DATA_PATH_DEFAULT)
    st.sidebar.write(f"Data: `{os.path.relpath(default_path, os.getcwd())}`")
    df = load_data(default_path)

    # Sidebar config
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"], index=1)
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    # Prepare features/target
    has_target = "Revenue" in df.columns
    if has_target:
        X, y = split_features_target(df)
        y = y.map(lambda v: 1 if str(v).strip().upper() in {"TRUE", "1", "YES"} else 0)
    else:
        X, y = df.copy(), None

    numeric, categorical = infer_column_types(X)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, None, None, None

    pipe = build_pipeline(model_name, numeric, categorical)
    with st.spinner("Training model..."):
        if y is not None:
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X, np.zeros(len(X)))
    st.success("Model ready.")

    # Tabs
    tab_about, tab_predict, tab_eval, tab_compare, tab_improve, tab_nb, tab_future = st.tabs(
        [
            "About",
            "Predict",
            "Train & Evaluate",
            "Compare Models",
            "Improve Model",
            "Notebook Results",
            "Future Work",
        ]
    )

    with tab_about:
        st.subheader("What This App Does")
        st.markdown(
            """
            <div class="callout">
            <span class="tag">Goal</span> Estimate the likelihood a session will result in a purchase and surface the drivers of conversion.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - Predict: fill session details (or pick a preset), set a decision threshold, and get BUY / NOT BUY.
            - Train & Evaluate: review metrics and diagnostic charts on a holdout set.
            - Compare Models: choose between Logistic Regression and Random Forest.
            - Improve Model: adjust key hyperparameters and observe metric changes.
            - Notebook Results and Details: learn the data and the methodology.
            """
        )
        st.subheader("About The Dataset")
        st.markdown(
            """
            - Source: UCI / Kaggle Online Shoppers Purchasing Intention Dataset  
              Link: https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset  
            - The dataset consists of feature vectors belonging to 12,330 sessions.  
            - Each session belongs to a different user within a 1‑year period to avoid bias toward a specific campaign, special day, user profile, or period.  
            - Content: 10 numerical and 8 categorical attributes.  
            - Class label: `Revenue` (whether a purchase occurred).
            """
        )
        st.subheader("ML Pipeline — How Predictions Are Made")
        with st.expander("See pipeline steps"):
            st.markdown(
                """
                1) Cleaning  
                   - Drop duplicates.  
                   - Impute missing values (numeric → median, categorical → most frequent).  
                2) Encoding  
                   - Ordinal-encode categorical features (`Month`, `VisitorType`, `Weekend`).  
                3) Scaling  
                   - Standardize all features to stabilize models and improve comparability.  
                4) Modeling  
                   - Logistic Regression (interpretable coefficients, linear decision boundary).  
                   - Random Forest (non-linear, robust, exposes feature importances).  
                5) Decision Threshold  
                   - Convert probability to BUY/NOT BUY using adjustable threshold (trade-off precision vs. recall).
                """
            )
        st.subheader("Why These Models?")
        st.markdown(
            """
            - Logistic Regression: fast, easy to interpret, solid baseline for linearly-separable signals.  
            - Random Forest: captures non-linear interactions and typically lifts recall/ROC AUC on tabular data.  
            Use Compare/Improve tabs to decide which best fits your goal (precision-led vs. recall-led).
            """
        )

    with tab_predict:
        st.subheader("Predict - Will the shopper buy?")
        st.caption("Fill session details or pick a preset; adjust threshold and predict.")

        presets = {
            "None": {},
            "New visitor - exploring many products": {
                "Administrative": 0, "Informational": 0, "ProductRelated": 25,
                "Administrative_Duration": 0.0, "Informational_Duration": 0.0, "ProductRelated_Duration": 1200.0,
                "BounceRates": 0.02, "ExitRates": 0.05, "PageValues": 10.0, "SpecialDay": 0.0,
                "Month": "Nov", "VisitorType": "New_Visitor", "Weekend": "No",
            },
            "Returning visitor - quick bounce": {
                "Administrative": 0, "Informational": 0, "ProductRelated": 1,
                "Administrative_Duration": 0.0, "Informational_Duration": 0.0, "ProductRelated_Duration": 5.0,
                "BounceRates": 0.8, "ExitRates": 0.8, "PageValues": 0.0, "SpecialDay": 0.0,
                "Month": "Feb", "VisitorType": "Returning_Visitor", "Weekend": "No",
            },
            "Deal day - high purchase intent": {
                "Administrative": 1, "Informational": 1, "ProductRelated": 15,
                "Administrative_Duration": 30.0, "Informational_Duration": 20.0, "ProductRelated_Duration": 900.0,
                "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 20.0, "SpecialDay": 0.8,
                "Month": "Nov", "VisitorType": "Returning_Visitor", "Weekend": "Yes",
            },
        }
        preset_name = st.selectbox("Preset scenario", list(presets.keys()), index=0)

        months_from_data = [str(m) for m in X["Month"].dropna().unique().tolist()] if "Month" in X.columns else []
        default_months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months = months_from_data or default_months

        # Defaults from data medians
        admin_def = int(X.get("Administrative", pd.Series([0])).median() or 0)
        info_def = int(X.get("Informational", pd.Series([0])).median() or 0)
        prod_def = int(X.get("ProductRelated", pd.Series([1])).median() or 1)
        admin_d_def = float(X.get("Administrative_Duration", pd.Series([0.0])).median() or 0.0)
        info_d_def = float(X.get("Informational_Duration", pd.Series([0.0])).median() or 0.0)
        prod_d_def = float(X.get("ProductRelated_Duration", pd.Series([0.0])).median() or 0.0)
        bounce_def = float(min(1.0, max(0.0, (X.get("BounceRates", pd.Series([0.05])).median() or 0.05))))
        exit_def = float(min(1.0, max(0.0, (X.get("ExitRates", pd.Series([0.1])).median() or 0.1))))
        page_val_def = float(X.get("PageValues", pd.Series([0.0])).median() or 0.0)
        special_def = float(min(1.0, max(0.0, (X.get("SpecialDay", pd.Series([0.0])).median() or 0.0))))
        month_def = months[0] if months else "Feb"
        vtype_ui_def = "Returning"
        weekend_choice_def = "No"

        # Apply preset overrides
        if preset_name != "None":
            p = presets[preset_name]
            admin_def = p.get("Administrative", admin_def)
            info_def = p.get("Informational", info_def)
            prod_def = p.get("ProductRelated", prod_def)
            admin_d_def = p.get("Administrative_Duration", admin_d_def)
            info_d_def = p.get("Informational_Duration", info_d_def)
            prod_d_def = p.get("ProductRelated_Duration", prod_d_def)
            bounce_def = p.get("BounceRates", bounce_def)
            exit_def = p.get("ExitRates", exit_def)
            page_val_def = p.get("PageValues", page_val_def)
            special_def = p.get("SpecialDay", special_def)
            month_def = p.get("Month", month_def)
            vtype_ui_def = {"Returning_Visitor": "Returning", "New_Visitor": "New", "Other": "Other"}.get(p.get("VisitorType", "Returning_Visitor"), vtype_ui_def)
            weekend_choice_def = p.get("Weekend", weekend_choice_def)

        with st.form(key="predict_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                admin = st.number_input("Admin pages visited", min_value=0, value=int(admin_def), help="Number of account/help pages viewed.")
                info = st.number_input("Info pages visited", min_value=0, value=int(info_def), help="Number of information pages viewed (policies, FAQ, etc.).")
                prod = st.number_input("Product pages visited", min_value=0, value=int(prod_def), help="How many product pages were viewed.")

            with col2:
                admin_d = st.number_input("Time on admin pages (sec)", min_value=0.0, value=float(admin_d_def), help="Total time spent on account/help pages.")
                info_d = st.number_input("Time on info pages (sec)", min_value=0.0, value=float(info_d_def), help="Total time spent on information pages.")
                prod_d = st.number_input("Time on product pages (sec)", min_value=0.0, value=float(prod_d_def), help="Total time spent viewing products.")

            with col3:
                bounce = st.number_input("Immediate exits (0-1)", min_value=0.0, max_value=1.0, value=float(bounce_def), help="Higher means more sessions ended right away.")
                exit_r = st.number_input("Exit rate (0-1)", min_value=0.0, max_value=1.0, value=float(exit_def), help="Fraction of page views that ended the session.")
                page_val = st.number_input("Average page value", min_value=0.0, value=float(page_val_def), help="Higher when pages are near conversion steps.")

            col4, col5, col6, col7 = st.columns(4)
            with col4:
                special = st.number_input("Near special day (0-1)", min_value=0.0, max_value=1.0, value=float(special_def), help="Closer to 1 if session is near a special day (e.g., holidays).")
                month = st.selectbox("Month of visit", options=months, index=(months.index(month_def) if month_def in months else 0), help="Month when the session occurred.")

            os_default = int(X.get("OperatingSystems", pd.Series([1])).median() or 1)
            browser_default = int(X.get("Browser", pd.Series([1])).median() or 1)
            region_default = int(X.get("Region", pd.Series([1])).median() or 1)
            traffic_default = int(X.get("TrafficType", pd.Series([1])).median() or 1)
            show_adv = st.checkbox("Show advanced technical fields (codes)", value=False)
            if show_adv:
                with col5:
                    os_val = st.number_input("Operating system (code)", min_value=1, value=os_default, help="Encoded OS identifier from dataset.")
                    browser = st.number_input("Browser (code)", min_value=1, value=browser_default, help="Encoded browser identifier.")
                with col6:
                    region = st.number_input("Region (code)", min_value=1, value=region_default, help="Encoded region identifier.")
                    traffic = st.number_input("Traffic source (code)", min_value=1, value=traffic_default, help="Encoded traffic source identifier.")
            else:
                os_val = os_default
                browser = browser_default
                region = region_default
                traffic = traffic_default

            with col7:
                vtype_ui = st.selectbox("Visitor type", options=["Returning", "New", "Other"], index=( ["Returning", "New", "Other"].index(vtype_ui_def) if vtype_ui_def in ["Returning", "New", "Other"] else 0 ), help="Is this a returning or new visitor?")
                weekend_choice = st.selectbox("Weekend session?", options=["No", "Yes"], index=( ["No", "Yes"].index(weekend_choice_def) if weekend_choice_def in ["No", "Yes"] else 0 ), help="Whether the session was on a weekend.")

            threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, help="Classify as BUY when predicted probability is at or above this value.")
            submit = st.form_submit_button("Predict")

        if submit:
            vtype = {"Returning": "Returning_Visitor", "New": "New_Visitor", "Other": "Other"}.get(vtype_ui, "Returning_Visitor")
            weekend = "TRUE" if weekend_choice == "Yes" else "FALSE"
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

    with tab_eval:
        st.subheader("Evaluation")
        if y is not None:
            y_pred = pipe.predict(X_test)
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
            with st.expander("Why these metrics?"):
                st.markdown("- Accuracy: overall correctness; can be misleading with imbalance.")
                st.markdown("- Precision: of predicted BUYs, how many were true BUYs (cost of false positives).")
                st.markdown("- Recall: of actual BUYs, how many we caught (cost of missed buyers).")
                st.markdown("- F1: balance between precision and recall.")
                st.markdown("- ROC AUC: ranking quality across thresholds.")

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
            # Show importance for RF or abs coef for LR, with simplified labels
            model = pipe.named_steps["model"]
            prep = pipe.named_steps.get("prep")
            try:
                raw_names = list(prep.get_feature_names_out()) if prep is not None else None
            except Exception:
                raw_names = None

            # Map raw names like 'num__ExitRates' or 'cat__VisitorType' to simplified, human labels
            friendly_map = {
                "Administrative": "Admin pages visited",
                "Administrative_Duration": "Time on admin pages (sec)",
                "Informational": "Info pages visited",
                "Informational_Duration": "Time on info pages (sec)",
                "ProductRelated": "Product pages visited",
                "ProductRelated_Duration": "Time on product pages (sec)",
                "BounceRates": "Immediate exits (0-1)",
                "ExitRates": "Exit rate (0-1)",
                "PageValues": "Average page value",
                "SpecialDay": "Near special day (0-1)",
                "Month": "Month",
                "OperatingSystems": "OS (code)",
                "Browser": "Browser (code)",
                "Region": "Region (code)",
                "TrafficType": "Traffic source (code)",
                "VisitorType": "Visitor type",
                "Weekend": "Weekend session",
            }

            def simplify(n: str) -> str:
                base = n.split("__")[-1] if "__" in n else n
                return friendly_map.get(base, base)

            try:
                if hasattr(model, "feature_importances_"):
                    vals = model.feature_importances_
                elif hasattr(model, "coef_"):
                    vals = np.abs(model.coef_)[0]
                else:
                    vals = None
            except Exception:
                vals = None
            if vals is None:
                st.info("Model does not expose feature importances.")
            else:
                names = [simplify(n) for n in raw_names] if raw_names is not None else [f"f{i}" for i in range(len(vals))]
                order = np.argsort(vals)[::-1][:15]
                imp_df = pd.DataFrame({"Feature": np.array(names)[order], "Importance": vals[order]})
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.barplot(data=imp_df, y="Feature", x="Importance", palette="viridis", ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)
        else:
            st.info("No target column detected; evaluation metrics are hidden.")

    with tab_compare:
        st.subheader("Model Comparison")
        st.caption("Trains Logistic Regression and Random Forest on the same split and compares metrics.")
        if y is not None:
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

    with tab_improve:
        st.subheader("Improve Model")
        st.caption("Tune key hyperparameters and see how metrics change on the same train/test split.")
        if y is not None:
            choice = st.selectbox("Choose model to tune", ["Logistic Regression", "Random Forest"], index=0)
            if choice == "Logistic Regression":
                c1, c2, c3 = st.columns(3)
                C = c1.number_input("C (inverse reg)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
                max_iter = c2.number_input("Max iterations", min_value=100, max_value=5000, value=1000, step=100)
                cw = c3.selectbox("Class weight", ["None", "balanced"], index=0)
                model = LogisticRegression(C=C, max_iter=int(max_iter), class_weight=(None if cw == "None" else "balanced"))
            else:
                c1, c2, c3, c4 = st.columns(4)
                n_estimators = c1.slider("Trees", 50, 1000, 300, 50)
                max_depth = c2.selectbox("Max depth", ["None", 5, 10, 20, 30], index=0)
                min_samples_leaf = c3.selectbox("Min samples leaf", [1, 2, 4, 8], index=0)
                cw = c4.selectbox("Class weight", ["None", "balanced"], index=0)
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=(None if max_depth == "None" else int(max_depth)),
                    min_samples_leaf=int(min_samples_leaf),
                    class_weight=(None if cw == "None" else "balanced"),
                    random_state=42,
                )

            pipe_tuned = Pipeline(steps=[
                ("prep", ColumnTransformer([
                    ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), numeric),
                    ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
                ], remainder="drop")),
                ("scale", StandardScaler()),
                ("model", model),
            ])
            with st.spinner("Training tuned model..."):
                pipe_tuned.fit(X_train, y_train)
            yp = pipe_tuned.predict(X_test)
            try:
                pr = pipe_tuned.predict_proba(X_test)[:, 1]
            except Exception:
                pr = None
            m = compute_metrics(y_test, yp, pr)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{m['accuracy']:.3f}")
            c2.metric("Precision", f"{m['precision']:.3f}")
            c3.metric("Recall", f"{m['recall']:.3f}")
            c1.metric("F1", f"{m['f1']:.3f}")
            if m.get("roc_auc") is not None and not np.isnan(m.get("roc_auc", np.nan)):
                c2.metric("ROC AUC", f"{m['roc_auc']:.3f}")
            st.caption("These results use the same train/test split as the Evaluate tab.")
        else:
            st.info("Metrics unavailable without the 'Revenue' target column.")

    with tab_nb:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Rows: {len(X)} | Numeric: {len(numeric)} | Categorical: {len(categorical)}")
        if st.checkbox("Show cleaned preview (drop duplicates)"):
            dfc = nb_style_clean_preview(df)
            st.write(f"Rows after de-duplication: {len(dfc)}")
            st.dataframe(dfc.head(20), use_container_width=True)

        st.subheader("Notebook Results - Key Insights")
        st.caption("Replicates core visuals from the notebook with explanations.")
        if y is not None:
            plot_class_balance(y)
            st.markdown("Class balance impacts metric choice and threshold tuning.")
        st.markdown("---")
        st.subheader("Correlation Heatmap")
        plot_corr_heatmap(df)
        st.markdown("Correlations highlight linear relationships; strong ones can inform features.")
        st.markdown("---")
        st.subheader("Distributions of Key Features")
        feat = st.selectbox(
            "Choose a feature",
            [c for c in ["ProductRelated_Duration", "ProductRelated", "PageValues", "BounceRates", "ExitRates", "SpecialDay"] if c in df.columns],
        )
        if feat:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df[feat], kde=True, ax=ax, color="#4c78a8")
            ax.set_title(f"Distribution: {feat}")
            st.pyplot(fig)

    

    with tab_future:
        st.subheader("Future Work")
        st.markdown(
            """
            - Feature engineering: interaction terms, recency/frequency, time-based signals.
            - Handle imbalance: class_weight, SMOTE, or threshold optimization.
            - Hyperparameter tuning: systematic search; try gradient boosting.
            - Probability calibration: Platt scaling or isotonic regression.
            - Explainability: SHAP values for single predictions and global insights.
            - Robust evaluation: time-aware splits, cross-validation, confidence intervals.
            - Deployment: persist pipeline, serve via API, monitor drift and performance.
            """
        )


if __name__ == "__main__":
    main()
