import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Fix imports
sys.path.append(os.path.abspath(".."))

# Import your modules
from data_cleaning import load_and_clean
from xg_model import train_xgb_model

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# --- Page Config ---
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("🏦 Customer Churn Analytics Dashboard")

# --- Load Data ---
df = load_and_clean("data/European_Bank.csv")

# --- Train Model ---
model, X_test, y_test, y_pred, y_prob = train_xgb_model(df)

# =========================================================
# 🔎 FILTER SECTION
# =========================================================
st.sidebar.header("🔎 Filters")

geo_options = ["All"] + list(df['Geography'].unique())
selected_geo = st.sidebar.selectbox("Geography", geo_options)

gender_options = ["All"] + list(df['Gender'].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

active_options = ["All"] + list(df['IsActiveMember'].unique())
selected_active = st.sidebar.selectbox("Active Member", active_options)

age_range = st.sidebar.slider(
    "Age Range",
    int(df['Age'].min()),
    int(df['Age'].max()),
    (25, 60)
)

# =========================================================
# 🔥 APPLY FILTER LOGIC
# =========================================================
filtered_df = df.copy()

if selected_geo != "All":
    filtered_df = filtered_df[filtered_df['Geography'] == selected_geo]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if selected_active != "All":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == selected_active]

filtered_df = filtered_df[
    filtered_df['Age'].between(age_range[0], age_range[1])
]

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# Show filters
st.sidebar.markdown("---")
st.sidebar.write("### Active Filters")
st.sidebar.write(f"Geography: {selected_geo}")
st.sidebar.write(f"Gender: {selected_gender}")
st.sidebar.write(f"Active: {selected_active}")

# =========================================================
# 📌 KPI SECTION
# =========================================================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Rate", f"{filtered_df['Exited'].mean():.2%}")

with col2:
    st.metric("Customers", len(filtered_df))

with col3:
    st.metric("Avg Balance", f"{filtered_df['Balance'].mean():,.0f}")

# =========================================================
# 🌍 CHURN BY GEOGRAPHY
# =========================================================
st.subheader("🌍 Churn by Geography")

geo_churn = filtered_df.groupby('Geography')['Exited'].mean().reset_index()

fig1 = px.bar(
    geo_churn,
    x='Geography',
    y='Exited',
    color='Geography',
    title="Churn Rate by Country"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# 👥 AGE VS CHURN
# =========================================================
st.subheader("👥 Age Distribution vs Churn")

fig2 = px.histogram(
    filtered_df,
    x='Age',
    color='Exited',
    nbins=30,
    barmode='overlay'
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 📊 FEATURE IMPORTANCE
# =========================================================
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
features = X_test.columns

fi_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig3 = px.bar(
    fi_df.head(10),
    x='Importance',
    y='Feature',
    orientation='h',
    title="Top Drivers of Churn"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# 📈 MODEL PERFORMANCE (VISUAL UPGRADE)
# =========================================================
st.subheader("📈 Model Performance")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)

# KPI cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Recall (Churn Detection)", f"{recall:.2%}")

with col3:
    st.metric("Precision", f"{precision:.2%}")

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.style.format("{:.2f}"))

# Business Insight
st.subheader("💡 Model Insight")

st.info(f"""
- Model detects {recall:.0%} of churn customers  
- {fn} churn customers are missed (false negatives)  
- Precision indicates reliability of churn predictions  
""")

# =========================================================
# 🎯 PREDICTION TOOL
# =========================================================
st.subheader("🎯 Predict Customer Churn")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 35)
    balance = st.number_input("Balance", 0, 250000, 50000)
    credit = st.slider("Credit Score", 300, 900, 600)

with col2:
    products = st.selectbox("Products", [1, 2, 3, 4])
    active = st.selectbox("Active Member", [0, 1])
    geo = st.selectbox("Geography", df['Geography'].unique())

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Balance": balance,
        "CreditScore": credit,
        "NumOfProducts": products,
        "IsActiveMember": active,
        "Geography": geo
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_test.columns, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]

    if prob > 0.5:
        st.error(f"⚠️ High Churn Risk ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk ({prob:.2%})")

# =========================================================
# 💡 FINAL INSIGHTS
# =========================================================
st.subheader("💡 Key Insights")

st.markdown("""
- Inactive customers show significantly higher churn  
- Geography impacts churn behavior  
- Customers with fewer products are more likely to leave  
- Balance and credit score are strong predictors  
""")