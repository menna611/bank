# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.title("🧠 Interactive ML : DT, RF & XGBoost")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data_train.csv")
    return df

df = load_data()

if st.checkbox("Show raw data"):
    st.write(df.head())

# Drop unnecessary or high-null columns
df = df.drop(columns=["SK_ID_CURR"], errors="ignore")
df = df.drop(columns=["OCCUPATION_TYPE"], errors="ignore")  # too many NaNs

# Fill missing values
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop("TARGET", axis=1)
y = df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar: Model Selection
st.sidebar.header("⚙️ Model Selection & Parameters")
model_name = st.sidebar.selectbox("Choose Model", ["Decision Tree", "Random Forest", "XGBoost"])

# Set hyperparameters for each model
if model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 30, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

elif model_name == "XGBoost":
    n_estimators = st.sidebar.slider("Number of Boosting Rounds", 10, 200, 100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                          use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance
st.subheader("📊 Model Performance")
st.write(f"**Model**: {model_name}")
st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Predict custom input
st.subheader("🔍 Predict with Custom Input")
user_input = {}
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"🔮 Predicted TARGET: {prediction}")
