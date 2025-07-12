# Logistic Regression on Titanic Dataset with Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction - Logistic Regression")

# Load datasets
@st.cache_data
def load_data():
    train = pd.read_csv("Titanic_train.csv")
    test = pd.read_csv("Titanic_test.csv")
    return train, test

train_df, test_df = load_data()
df = train_df.copy()

# EDA Section
st.header("🔍 Exploratory Data Analysis")
if st.checkbox("Show raw data"):
    st.write(df.head())

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.header("📊 Model Evaluation Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
with col2:
    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.2f}")

# Confusion matrix
st.subheader("📌 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label="ROC Curve")
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic")
ax2.legend()
st.pyplot(fig2)

# Prediction Interface
st.header("🎯 Predict Survival")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [0 if sex == "male" else 1],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [le.fit(df['Embarked']).transform([embarked])[0]]
})

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Survived' if pred == 1 else 'Did Not Survive'} ({prob:.2%} probability)")

