import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "telco_rf_pipeline.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "telco_metrics.joblib")

st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")
st.title("Telco Customer Churn Dashboard")
st.caption("Clasificación supervisada con Streamlit + Scikit-learn")


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).replace({"0": "No", "1": "Yes", 0: "No", 1: "Yes"})

    return df


@st.cache_data(show_spinner=False)
def load_telco_data():
    df = pd.read_csv(DATA_URL)
    df = clean_telco(df)
    return df


def split_columns(df: pd.DataFrame, target: str = "Churn"):
    X = df.drop(columns=[target])
    y = df[target].map({"No": 0, "Yes": 1}).astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=mutual_info_classif, k=20)),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    return pipeline


@st.cache_resource(show_spinner=True)
def train_or_load_model():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH)

    df = load_telco_data()
    X, y, numeric_features, categorical_features = split_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "selector__k": [15, 20, 25, "all"],
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": grid.best_params_,
        "cv_best_f1": float(grid.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_names_raw": X.columns.tolist(),
        "training_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return best_model, metrics


def get_feature_importance(best_model):
    preprocessor = best_model.named_steps["preprocessor"]
    selector = best_model.named_steps["selector"]
    model = best_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    support = selector.get_support()
    selected_feature_names = feature_names[support]
    importances = model.feature_importances_

    fi = pd.DataFrame(
        {"feature": selected_feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return fi


def get_expected_columns():
    df = load_telco_data()
    cols = [c for c in df.columns if c != "Churn"]
    return cols


def prepare_batch_input(df_batch: pd.DataFrame):
    df_batch = clean_telco(df_batch)
    expected_cols = get_expected_columns()
    missing = [c for c in expected_cols if c not in df_batch.columns]
    extra = [c for c in df_batch.columns if c not in expected_cols]

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if extra:
        df_batch = df_batch.drop(columns=extra)

    return df_batch[expected_cols]


model, metrics = train_or_load_model()
feature_importance = get_feature_importance(model)

with st.sidebar:
    st.header("Resumen del modelo")
    st.metric("CV F1 (mejor)", f"{metrics['cv_best_f1']:.3f}")
    st.metric("Test F1", f"{metrics['test_f1']:.3f}")
    st.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
    st.write("**Mejores hiperparámetros**")
    st.json(metrics["best_params"])

tab1, tab2, tab3 = st.tabs(["Predicción individual", "Predicción por lote (CSV)", "Métricas e importancia"])

with tab1:
    st.subheader("Predicción individual")
    st.write("Completa el formulario para estimar la probabilidad de churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure", min_value=0, max_value=72, value=12)

    with col2:
        phone = st.selectbox("PhoneService", ["No", "Yes"])
        multiple_lines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])

    with col3:
        device_prot = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("PaperlessBilling", ["No", "Yes"])
        payment = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    col4, col5 = st.columns(2)
    with col4:
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    with col5:
        total = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=850.0, step=0.1)

    if st.button("Predecir churn", type="primary"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prob = model.predict_proba(input_df)[:, 1][0]
        pred = int(model.predict(input_df)[0])
        label = "Yes" if pred == 1 else "No"

        st.success(f"Predicción de churn: **{label}**")
        st.metric("Probabilidad de churn", f"{prob:.2%}")

        st.write("Interpretación rápida:")
        if prob >= 0.7:
            st.warning("Riesgo alto de churn. Conviene evaluar acciones de retención.")
        elif prob >= 0.4:
            st.info("Riesgo medio de churn. Hay señales de posible abandono.")
        else:
            st.info("Riesgo bajo de churn.")

with tab2:
    st.subheader("Predicción por lote")
    st.write("Sube un CSV con las columnas del dataset Telco, excepto `Churn`.")

    expected = get_expected_columns()
    with st.expander("Ver columnas esperadas"):
        st.write(expected)

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    template_df = load_telco_data().drop(columns=["Churn"]).head(5)
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar plantilla CSV",
        data=csv_template,
        file_name="telco_batch_template.csv",
        mime="text/csv"
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Vista previa del archivo cargado:")
        st.dataframe(batch_df.head())

        try:
            prepared = prepare_batch_input(batch_df)
            batch_prob = model.predict_proba(prepared)[:, 1]
            batch_pred = model.predict(prepared)

            results = prepared.copy()
            results["churn_probability"] = batch_prob
            results["churn_prediction"] = np.where(batch_pred == 1, "Yes", "No")

            st.success("Predicciones generadas correctamente.")
            st.dataframe(results.head(20))

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar predicciones",
                data=csv_out,
                file_name="telco_batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"No fue posible generar las predicciones: {e}")

with tab3:
    st.subheader("Desempeño del modelo en test")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['test_precision']:.3f}")
    c3.metric("Recall", f"{metrics['test_recall']:.3f}")
    c4.metric("F1", f"{metrics['test_f1']:.3f}")
    c5.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")

    st.write("**Matriz de confusión**")
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Real No", "Real Yes"], columns=["Pred No", "Pred Yes"])
    st.dataframe(cm_df)

    st.write("**Feature Importance**")
    top_n = st.slider("Número de variables a mostrar", min_value=5, max_value=20, value=12)
    top_fi = feature_importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_fi["feature"], top_fi["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top variables del Random Forest")
    st.pyplot(fig)

    st.write("**Top variables**")
    st.dataframe(feature_importance.head(20), use_container_width=True)

st.markdown("---")
st.caption("Aplicación construida para el reto de clasificación con Telco Customer Churn.")
