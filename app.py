import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AutoScope", layout="wide")

CURRENT_YEAR = 2026
FEATURES = ["price", "age", "mileage", "tax", "mpg", "engineSize"]


@st.cache_resource
def load_models():
    price_model = joblib.load("price_model.pkl")
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
    return price_model, kmeans, scaler


@st.cache_data
def load_data():
    cars = pd.read_csv("cars_clean.csv")
    clusters = pd.read_csv("clusters.csv")
    scores = pd.read_csv("scores.csv")
    return cars, clusters, scores


price_model, kmeans, scaler = load_models()
cars, clusters, scores = load_data()


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(price_model["model"])


@st.cache_data
def get_global_shap():
    sample = cars.sample(n=min(300, len(cars)), random_state=42)
    X = sample[["brand", "age", "mileage", "transmission", "fuelType", "tax", "mpg", "engineSize"]]
    X = pd.get_dummies(X).reindex(columns=price_model["columns"], fill_value=0).astype(float)
    return get_explainer()(X)


# ------------------------------------------------------------ PAGE 1: PRICE PREDICTION
def prediction_page():
    st.title("Car Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", sorted(cars["brand"].unique()))
        year = st.number_input("Year", 1990, CURRENT_YEAR, 2018)
        mileage = st.number_input("Mileage", 0, 300000, 30000)
        transmission = st.selectbox("Transmission", sorted(cars["transmission"].unique()))
    with col2:
        fuel = st.selectbox("Fuel Type", sorted(cars["fuelType"].unique()))
        tax = st.number_input("Tax (£)", 0, 600, 150)
        mpg = st.number_input("MPG", 5.0, 200.0, 50.0)
        engine = st.number_input("Engine Size", 0.5, 7.0, 1.6, step=0.1)

    if st.button("Predict Price"):
        age = CURRENT_YEAR - year
        row = pd.DataFrame([{
            "brand": brand, "age": age, "mileage": mileage,
            "transmission": transmission, "fuelType": fuel,
            "tax": tax, "mpg": mpg, "engineSize": engine,
        }])
        row = pd.get_dummies(row).reindex(columns=price_model["columns"], fill_value=0)
        price = price_model["model"].predict(row)[0]

        seg_input = pd.DataFrame([[price, age, mileage, tax, mpg, engine]], columns=FEATURES)
        segment = kmeans.predict(scaler.transform(seg_input))[0]

        st.metric("Predicted Price", f"£{price:,.0f}")
        st.metric("Segment (K-Means)", f"Cluster {segment}")

        st.subheader("Segment Profile")
        profile = clusters[clusters["KMeans"] == segment][FEATURES].mean().round(1)
        st.dataframe(profile.to_frame("Average"))
        st.write("Top brands:", ", ".join(clusters[clusters["KMeans"] == segment]["brand"].value_counts().head(3).index))
        st.write("Top fuel type:", clusters[clusters["KMeans"] == segment]["fuelType"].mode()[0])

        st.subheader("Why this price? (SHAP)")
        shap_values = get_explainer()(row.astype(float))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
        plt.close()


# ------------------------------------------------------------ PAGE 2: CLUSTERING
def clustering_page():
    st.title("Market Clustering")

    col1, col2 = st.columns(2)
    algo = col1.selectbox("Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])
    view = col2.radio("View", ["PCA", "t-SNE"], horizontal=True)

    x, y = ("pca1", "pca2") if view == "PCA" else ("tsne1", "tsne2")
    plot_df = clusters.copy()
    plot_df["Cluster"] = plot_df[algo].astype(str)

    fig = px.scatter(plot_df, x=x, y=y, color="Cluster",
                     hover_data=["brand", "price", "age", "mileage"], height=550)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Silhouette Scores")
    st.dataframe(scores, hide_index=True)

    st.subheader(f"{algo} Cluster Profile")
    profile = plot_df.groupby(algo)[FEATURES].mean().round(1)
    profile["count"] = plot_df.groupby(algo).size()
    profile["top_brand"] = plot_df.groupby(algo)["brand"].agg(lambda s: s.mode()[0])
    profile["top_fuel"] = plot_df.groupby(algo)["fuelType"].agg(lambda s: s.mode()[0])
    st.dataframe(profile)


# ------------------------------------------------------------ PAGE 3: SHAP
def shap_page():
    st.title("Model Explainability (SHAP)")
    shap_values = get_global_shap()

    st.subheader("Feature Importance")
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Feature Impact on Price")
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.close()


# ------------------------------------------------------------ SIDEBAR MENU
with st.sidebar:
    page = option_menu(
        "AutoScope",
        ["Price Prediction", "Clustering", "SHAP"],
        icons=["currency-pound", "diagram-3", "bar-chart"],
        default_index=0,
    )

if page == "Price Prediction":
    prediction_page()
elif page == "Clustering":
    clustering_page()
else:
    shap_page()
