# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.title("🧠 ML Models: Decision Tree, Random Forest & XGBoost")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data_train.csv")
    return df

df = load_data()

if st.checkbox("Show raw data"):
    st.write(df.head())

# Preprocessing
df = df.drop(columns=["SK_ID_CURR"], errors="ignore")
df = df.drop(columns=["OCCUPATION_TYPE"], errors="ignore")  # too many NaNs

# Fill missing numeric values
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Define features and target
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar options
st.sidebar.title("⚙️ Choose Model")
model_name = st.sidebar.selectbox("Select a model", ["Decision Tree", "Random Forest", "XGBoost"])

# Model selection
if model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
elif model_name == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📊 Model Performance")
st.write(f"**Selected Model**: {model_name}")
st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Interactive input
st.subheader("🔍 Predict with Custom Input")
user_input = {}
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Predicted TARGET: {prediction}")
