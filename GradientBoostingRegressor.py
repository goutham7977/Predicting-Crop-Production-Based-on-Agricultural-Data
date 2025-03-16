import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("FAOSTAT_TRANSFORMED_CLEANED.csv")

# Drop missing values
df = df.dropna(subset=['Production', 'Area harvested', 'Yield'])

# Select features and target variable
X = df[['Area harvested', 'Yield']]
y = df['Production']

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌾 Crop Production Prediction using Gradient Boosting")

# Sidebar Filters
st.sidebar.header("🔹 Select Inputs")

# Dropdowns for user selection
year = st.sidebar.selectbox("Select Year", df["Year"].unique())
item = st.sidebar.selectbox("Select Crop Item", df["Item"].unique())
area = st.sidebar.selectbox("Select Area (Country/Region)", df["Area"].unique())

# User input fields for numerical values
area_harvested = st.sidebar.number_input("Enter Area Harvested (Hectares)", min_value=0, value=int(df["Area harvested"].mean()))
yield_value = st.sidebar.number_input("Enter Yield (tons per hectare)", min_value=0.0, value=float(df["Yield"].mean()))

# Prediction Button
if st.sidebar.button("🔍 Predict Production"):
    input_data = pd.DataFrame({"Area harvested": [area_harvested], "Yield": [yield_value]})
    input_scaled = scaler.transform(input_data)  
    prediction = model.predict(input_scaled)
    
    st.success(f"🌱 Estimated Production for {item} in {area} ({year}): **{prediction[0]:,.2f} tons**")

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"**Mean Squared Error:** {mse:,.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Scatter Plot of Actual vs Predicted
st.subheader("📉 Actual vs Predicted Production")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax, color="red", label="Predicted")
sns.scatterplot(x=y_test, y=y_test, ax=ax, color="blue", label="Actual", alpha=0.5)
ax.set_xlabel("Actual Production")
ax.set_ylabel("Predicted Production")
ax.set_title("Gradient Boosting Regression: Actual vs Predicted")
st.pyplot(fig)
