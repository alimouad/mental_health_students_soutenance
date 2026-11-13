
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load project files
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/raw/student_mental_health.csv"
MODEL_PATH = ROOT / "models/best_mental_health_model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = "Mental_Health_Risk"
ID = "Student_ID"

FEATURES = df.drop(columns=[ID, TARGET]).columns.tolist()

# 2. Page configuration
st.set_page_config(page_title="MindScope")

st.title("MindScope")
st.write("Student Mental Health Risk Prediction")
st.warning(
    "Educational project only. Predictions are not medical diagnoses."
)

# 3. Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "About"]
)


# 4. Dashboard
if page == "Dashboard":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Features", len(FEATURES))
    col3.metric("Missing Values", df[FEATURES].isnull().sum().sum())

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Target Distribution**")
        st.bar_chart(df[TARGET].value_counts())

    with col2:
        st.write("**Dataset Preview**")
        st.dataframe(df.head())

    st.divider()

    st.header("Model Performance")

    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, predictions, average='macro'):.2%}")
    col3.metric("Recall", f"{recall_score(y_test, predictions, average='macro'):.2%}")
    col4.metric("F1-score", f"{f1_score(y_test, predictions, average='macro'):.2%}")

    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_
    )

    st.dataframe(matrix_df)

# 5. Prediction
elif page == "Prediction":

    st.header("Predict Student Risk")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(new_data.head())

        missing_columns = [
            col for col in FEATURES if col not in new_data.columns
        ]

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")

        else:
            predictions = model.predict(new_data[FEATURES])

            results = new_data.copy()
            results["Predicted_Risk"] = predictions

            st.subheader("Prediction Results")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "mindscope_predictions.csv",
                "text/csv"
            )

# 6. About
else:

    st.header("About MindScope")

    st.write("""
    Mental Health is an educational machine learning project
    that predicts student mental health risk levels.

    **Dataset:** Student Mental Health Risk Prediction

    **Models tested:**
    - Logistic Regression
    - SVM
    - Random Forest

    **Selected model:** Logistic Regression

    **Technologies:** Python, Pandas, Scikit-learn and Streamlit.
    """)

    st.info(
        "This application is a learning prototype, "
        "not a clinical diagnosis or medical tool."
    )