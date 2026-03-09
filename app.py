import streamlit as st
import pandas as pd
from model import train_model

st.set_page_config(page_title="MoMA Artwork App", page_icon="🎨", layout="wide")

st.title("🎨 MoMA Artwork Classification Dashboard")

# load dataset once
@st.cache_data
def load_data():
    df = pd.read_csv("Artworks.csv")
    return df

df = load_data()

# train model once
@st.cache_resource
def load_model():
    return train_model()

model, accuracy, columns = load_model()

st.sidebar.header("Artwork Information")

height = st.sidebar.slider("Height (cm)", 0.0, 300.0, 50.0)
width = st.sidebar.slider("Width (cm)", 0.0, 300.0, 50.0)

department = st.sidebar.selectbox(
    "Department",
    ["Architecture & Design","Drawings & Prints","Painting & Sculpture","Photography"]
)

# create input dataframe
input_data = pd.DataFrame({
    "Height (cm)": [height],
    "Width (cm)": [width]
})

for col in columns:
    if col.startswith("Department_"):
        input_data[col] = 1 if col == f"Department_{department}" else 0

input_data = input_data.reindex(columns=columns, fill_value=0)

if st.button("Predict Artwork Type"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Classification: {prediction[0]}")

st.sidebar.write("Model Accuracy:", round(accuracy,2))

st.divider()

# ======================
# CHARTS SECTION
# ======================

st.subheader("📊 Artwork Classification Distribution")

class_counts = df['Classification'].value_counts().head(10)

st.bar_chart(class_counts)

st.subheader("🎨 Department Distribution")

dept_counts = df['Department'].value_counts()

st.bar_chart(dept_counts)

st.subheader("📏 Artwork Size Analysis")

size_df = df[['Height (cm)','Width (cm)']].dropna().sample(3000)

st.scatter_chart(size_df)