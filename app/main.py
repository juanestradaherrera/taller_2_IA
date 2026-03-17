import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "telco_rf_pipeline.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "telco_metrics.joblib")
FI_PATH = os.path.join(ARTIFACT_DIR, "telco_feature_importance.csv")

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
        df["SeniorCitizen"] = (
            df["SeniorCitizen"]
            .astype(str)
            .replace({"0": "No", "1": "Yes", 0: "No", 1: "Yes"})
        )

    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. "
            "Debes generarlo primero desde el notebook y guardarlo en app/artifacts/."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    return joblib.load(METRICS_PATH)


@st.cache_data
def load_feature_importance():
    if not os.path.exists(FI_PATH):
        return None
    return pd.read_csv(FI_PATH)


def get_expected_columns():
    return [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]


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


model = load_model()
metrics = load_metrics()
feature_importance = load_feature_importance()

with st.sidebar:
    st.header("Resumen del modelo")
    if metrics is not None:
        st.metric("CV F1 (mejor)", f"{metrics['cv_best_f1']:.3f}")
        st.metric("Test F1", f"{metrics['test_f1']:.3f}")
        st.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
        st.write("**Mejores hiperparámetros**")
        st.json(metrics["best_params"])
    else:
        st.info("No se encontraron métricas precalculadas.")

tab1, tab2, tab3 = st.tabs(
    ["Predicción individual", "Predicción por lote (CSV)", "Métricas e importancia"]
)

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

    template_df = pd.DataFrame(columns=expected)
    csv_template = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar plantilla CSV",
        data=csv_template,
        file_name="telco_batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

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
    st.subheader("Métricas e importancia")

    if metrics is not None:
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

    if feature_importance is not None:
        st.write("**Feature Importance**")
        top_n = st.slider("Número de variables a mostrar", min_value=5, max_value=20, value=12)
        top_fi = feature_importance.head(top_n).iloc[::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_fi["feature"], top_fi["importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top variables del modelo")
        st.pyplot(fig)

        st.write("**Top variables**")
        st.dataframe(feature_importance.head(20), use_container_width=True)
    else:
        st.info("No se encontró el archivo de importancia de variables.")

st.markdown("---")
st.caption("Aplicación construida para el reto de clasificación con Telco Customer Churn.")
