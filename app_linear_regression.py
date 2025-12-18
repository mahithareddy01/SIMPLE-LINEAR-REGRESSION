import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error

#Page Configuration
st.set_page_config("Linear Regression",layout="centered")

#load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")



st.markdown("""
<div class="card">
            <h1>LINEAR REGRESSION</h1>
            <p>Predict <b> Tip amount</b> based on <b> Total Bill</b> using Linear Regression...</p>
</div>
            """, unsafe_allow_html=True)

#Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()

# Display Dataset Preview
st.markdown('<div class="card2"><b>DATASET PREVIEW:</b></div>', unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown("---")

#Prepare Data
X = df[["total_bill"]]
y = df["tip"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
#Evaluate Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#Visualize Results
st.markdown('<div class="card2"><B>PREDICTIONS vs ACTUAL:</B></div>', unsafe_allow_html=True)
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Tips')
plt.scatter(X_test, y_pred, color='red', label='Predicted Tips')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.legend()
st.pyplot(plt)
st.markdown("---")

#Display Metrics
st.markdown('<div class="card2"><b>MODEL PERFORMANCE METRICS:</b></div>', unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
    <div class="metric-box">
        <span>MSE</span>
        <strong>{:.2f}</strong>
    </div>
    <div class="metric-box">
        <span>RMSE</span>
        <strong>{:.2f}</strong>
    </div>
    <div class="metric-box">
        <span>R² Score</span>
        <strong>{:.2f}</strong>
    </div>
</div>
""".format(mse, rmse, r2), unsafe_allow_html=True)
st.markdown("---")

#intercept and coefficient
st.markdown('<div class="card2"><b>MODEL PARAMETERS:</b></div>', unsafe_allow_html=True)
st.write(f"Intercept: {model.intercept_:.2f}")
st.write(f"Coefficient for Total Bill: {model.coef_[0]:.2f}")
st.markdown("---")

#prediction
st.markdown('<div class="card2"><b>PREDICT TIP AMOUNT:</b></div>', unsafe_allow_html=True)
bill = st.slider("Total Bill Amount ($)", float(df["total_bill"].min()), float(df["total_bill"].max()), 30.0)
tip = model.predict(scaler.transform([[bill]]))[0]
st.markdown(f"<div class='prediction-box'>Predicted Tip Amount: <strong>${tip:.2f}</strong></div>", unsafe_allow_html=True)

