# streamlit run crop_fertilizer_dashboard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Crop_and_fertilizer_with_usage.csv")

data = load_data()

# Title on main page
st.title("🌾 Crop Fertilizer Usage Analysis Dashboard 🌟")

# Data exploration in sidebar
st.sidebar.header("📊 Data Exploration")
if st.sidebar.checkbox("Show Dataset"):
    st.write(data.head())
if st.sidebar.checkbox("Show Dataset Summary"):
    st.write(data.describe())

# Drop unnecessary column
data = data.drop(columns=["Link"])

# Process categorical columns with individual encoders
categorical_cols = ["District_Name", "Soil_color", "Crop", "Fertilizer"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Define target and features
X = data.drop(columns=["Fertilizer_Usage"])
y = data["Fertilizer_Usage"]

# Data split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Model evaluation in sidebar
st.sidebar.header("📈 Model Performance")
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
if st.sidebar.checkbox("Show Model Metrics"):
    st.subheader("🚀 Model Evaluation Metrics")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R-squared Score:** {r2:.2f}")

# Visualizations in sidebar
st.sidebar.header("📊 Visualizations")
if st.sidebar.checkbox("Correlation Heatmap"):
    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax, cbar_kws={'shrink': 0.8})
    st.pyplot(fig)

if st.sidebar.checkbox("Feature Distribution"):
    feature = st.selectbox("Select a Feature:", X.columns)
    st.subheader(f"📊 Distribution of {feature}")
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax, color="teal")
    ax.set_title(f"Distribution of {feature}", fontsize=16)
    st.pyplot(fig)

if st.sidebar.checkbox("Crop-wise Fertilizer Usage"):
    st.subheader("🌱 Crop-wise Average Fertilizer Usage")
    crop_names = encoders["Crop"].inverse_transform(data["Crop"].unique())
    crop_avg_usage = data.groupby("Crop")["Fertilizer_Usage"].mean()
    crop_avg_usage.index = crop_names
    fig, ax = plt.subplots(figsize=(10, 6))
    crop_avg_usage.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_ylabel("Average Fertilizer Usage", fontsize=14)
    ax.set_title("Crop-wise Fertilizer Usage", fontsize=16)
    st.pyplot(fig)

if st.sidebar.checkbox("Fertilizer Usage by Soil Color"):
    st.subheader("🌍 Fertilizer Usage by Soil Color")
    soil_color_names = encoders["Soil_color"].inverse_transform(data["Soil_color"].unique())
    soil_avg_usage = data.groupby("Soil_color")["Fertilizer_Usage"].mean()
    soil_avg_usage.index = soil_color_names
    fig, ax = plt.subplots(figsize=(8, 5))
    soil_avg_usage.plot(kind="bar", ax=ax, color="coral", edgecolor="black")
    ax.set_ylabel("Average Fertilizer Usage", fontsize=14)
    ax.set_title("Fertilizer Usage by Soil Color", fontsize=16)
    st.pyplot(fig)

if st.sidebar.checkbox("Fertilizer Usage Pie Chart"):
    st.subheader("🥧 Fertilizer Usage Proportion")
    fertilizer_counts = data["Fertilizer"].value_counts()
    fertilizer_labels = encoders["Fertilizer"].inverse_transform(fertilizer_counts.index)
    explode = [0.1 if i == 0 else 0 for i in range(len(fertilizer_labels))]
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        fertilizer_counts, 
        labels=fertilizer_labels, 
        autopct="%1.1f%%", 
        startangle=90, 
        colors=sns.color_palette("Set2"),
        explode=explode,
        textprops={'fontsize': 12}
    )
    ax.set_title("Fertilizer Usage Proportion", fontsize=18)
    ax.legend(wedges, fertilizer_labels, title="Fertilizers", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)

if st.sidebar.checkbox("Fertilizer Usage by District"):
    st.subheader("📍 Fertilizer Usage by District")
    district_names = encoders["District_Name"].inverse_transform(data["District_Name"].unique())
    district_avg_usage = data.groupby("District_Name")["Fertilizer_Usage"].mean()
    district_avg_usage.index = district_names
    fig, ax = plt.subplots(figsize=(12, 6))
    district_avg_usage.plot(kind="bar", ax=ax, color="lightgreen", edgecolor="black")
    ax.set_ylabel("Average Fertilizer Usage", fontsize=14)
    ax.set_xlabel("District", fontsize=14)
    ax.set_title("Fertilizer Usage by District", fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

if st.sidebar.checkbox("Fertilizer Usage Trends"):
    st.subheader("📈 Fertilizer Usage Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=data, x="Rainfall", y="Fertilizer_Usage", hue="Crop", palette="tab10", ax=ax)
    ax.set_title("Fertilizer Usage vs Rainfall", fontsize=16)
    ax.set_xlabel("Rainfall", fontsize=14)
    ax.set_ylabel("Fertilizer Usage", fontsize=14)
    st.pyplot(fig)

# Checkbox to toggle prediction form location
show_prediction_in_main = st.sidebar.checkbox("Show Prediction on Main Dashboard", value=False)

# Prediction Section
if show_prediction_in_main:
    st.markdown("## 🔮 Prediction")
    with st.form("prediction_form_main"):
        user_inputs = {}
        for col in X.columns:
            if col in categorical_cols:
                options = list(encoders[col].classes_)
                selected = st.selectbox(f"Select {col}", options, key=f"{col}_main")
                user_inputs[col] = encoders[col].transform([selected])[0]
            else:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                default_val = float(data[col].mean())
                user_inputs[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=default_val, key=f"{col}_main")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            input_array = np.array([user_inputs[col] for col in X.columns]).reshape(1, -1)
            predicted_usage = model.predict(scaler.transform(input_array))[0]
            st.write(f"🌱 **Predicted Fertilizer Usage:** {predicted_usage:.2f}")
else:
    st.sidebar.header("🌟 Make a Prediction")
    with st.sidebar.form("prediction_form_sidebar"):
        user_inputs = {}
        for col in X.columns:
            if col in categorical_cols:
                options = list(encoders[col].classes_)
                selected = st.selectbox(f"Select {col}", options, key=f"{col}_sidebar")
                user_inputs[col] = encoders[col].transform([selected])[0]
            else:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                default_val = float(data[col].mean())
                user_inputs[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=default_val, key=f"{col}_sidebar")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            input_array = np.array([user_inputs[col] for col in X.columns]).reshape(1, -1)
            predicted_usage = model.predict(scaler.transform(input_array))[0]
            st.sidebar.write(f"🌱 **Predicted Fertilizer Usage:** {predicted_usage:.2f}")

st.write("🎉 Explore the relationships between crops, fertilizers, and soil properties interactively!")

# Footer section on main page
st.markdown("---")
st.markdown(
    '''
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <div style="flex: 1;">
            <h4 style="margin: 0; color: #6c757d;">👥 Team Members</h4>
            <p style="margin: 0;">
                <a href="https://www.linkedin.com/in/manasi-ahire-9ba112289" target="_blank" style="text-decoration: none; color: #007bff;">Manasi Ahire</a> |
                <a href="https://github.com/creator1" target="_blank" style="text-decoration: none; color: #24292e;">GitHub</a>
            </p>
            <p style="margin: 0;">
                <a href="https://www.linkedin.com/in/tanvi-aher-342758298" target="_blank" style="text-decoration: none; color: #007bff;">Tanavi Aher</a> |
                <a href="https://github.com/creator2" target="_blank" style="text-decoration: none; color: #24292e;">GitHub</a>
            </p>
            <p style="margin: 0;">
                <a href="https://www.linkedin.com/in/vedant-nighot-694325238" target="_blank" style="text-decoration: none; color: #007bff;">Vedant Nighot</a> |
                <a href="https://github.com/VedantNighot" target="_blank" style="text-decoration: none; color: #24292e;">GitHub</a>
            </p>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)
