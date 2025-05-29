
import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
import pandas as pd
import plotly.express as px

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Milk Prediction & Visualization", layout="wide")
st.title("🐄 Milk & Livestock Insights")

# -------------------- Mode Selection --------------------
mode = st.sidebar.radio("Choose Mode", ["Predict Milk Sale", "Explore Data"])

# -------------------- Spark Session --------------------
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("MilkApp") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
        .getOrCreate()

spark = get_spark_session()

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df_spark = spark.read.csv("hdfs://namenode:8020/Data/clean.csv", header=True, inferSchema=True)
    return df_spark.toPandas()

df = load_data()

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return PipelineModel.load("hdfs://namenode:8020/Data/models/rf_milk_model")

model = load_model()

# -------------------- Define Features --------------------
auto_features = [
    'No_death_B_Swiss_pure', 'No_death_Sheep', 'Milching_Jersey_cross', 'fish type:Mrigal',
    'No_death_Pig', 'other livestock:Cat', 'Prod type:Cheese', 'Bullock_Jatsha',
    'Calf_female_Jersey_cross', 'livestock type:Jersey Pure', 'l_female_Equine',
    'Calf_male_Jersey_cross', 'im_female_Pig', 'Milching_Doethra', 'l_female_Poultry',
    'Calf_female_Jersey_pure', 'No_death_Mithun', 'No_death_Zo', 'No_death_Cat',
    'im_male_Sheep', 'fish type:Cattla', 'fish type:Grass Carp', 'No_death_B_Swiss_cross',
    'Dry_Doethra', 'livestock type:Buffalo', 'Prod type:Chugo', 'Male progeny',
    'livestock type:Yanku-Yankum', 'livestock type:Jatsha-Jatsham_1',
    'livestock type:Nublang-Thrabum', 'im_female_Sheep', 'Milching_Jersey_pure',
    'No_death_Equine', 'livestock type:Brown Swiss Cross', 'Brd_bull_Jersey_cross',
    'qty_processed_milk', 'livestock type:Jaba_1', 'No_death_Buffalo',
    'Calf_y/n_Brown_swiss_cross_vec', 'prod yesNo_vec', 'Calf_y/n_Doeb_vec',
    'Calf_y/n_Mithun_vec', 'beehives yesNo_vec', 'meat yesNo_vec', 'Area_vec',
    'Did you or your household avail the AI services for your cattles?_vec',
    'Calf_y/n_Jatsha_vec', 'Dzongkhag_vec'
]

categorical_options = {
    "Dzongkhag": [
        "Bumthang", "Chhukha", "Dagana", "Gasa", "Haa", "Lhuntse", "Mongar", "Paro",
        "Pemagatshel", "Punakha", "Samdrup Jongkhar", "Samtse", "Sarpang", "Thimphu",
        "Trashigang", "Trashiyangtse", "Trongsa", "Tsirang", "Wangdue Phodrang",
        "Zhemgang", "unknown"
    ],
    "Area": ["Rural", "Urban", "unknown"],
    "Did you or your household avail the AI services for your cattles?": ["Yes", "No", "unknown"],
    "prod yesNo": ["Yes", "No", "unknown"],
    "Calf_y/n_Brown_swiss_cross": ["Yes", "No", "unknown"],
    "Calf_y/n_Doeb": ["Yes", "No", "unknown"],
    "Calf_y/n_Mithun": ["Yes", "No", "unknown"],
    "beehives yesNo": ["Yes", "No", "unknown"],
    "meat yesNo": ["Yes", "No", "unknown"],
    "Calf_y/n_Jatsha": ["Yes", "No", "unknown"]
}

# -------------------- Predict Mode --------------------
if mode == "Predict Milk Sale":
    st.header("Predict Milk Quantity Sold by Dzonkhag")

    input_dict = {}

    # Base Features
    # st.markdown("### Milk Production & Processing")
    base_numeric_features = [
        "qty_prod_milk", "qty_consumed_milk", "qty_processed_milk",
        "qty_sold_butter", "qty_prod_Chicken", "qty_prod_Fish", "qty_prod_Yakmeat"
    ]
    for feature in base_numeric_features:
        input_dict[feature] = st.number_input(feature.replace("_", " ").capitalize(), min_value=0.0)

    # Auto Features
    st.markdown("### Livestock, Breed, and Farm Details")
    for feature in auto_features:
        if feature.endswith("_vec"):
            raw = feature.replace("_vec", "")
            if raw in categorical_options:
                input_dict[raw] = st.selectbox(f"{raw}", categorical_options[raw])
            else:
                input_dict[raw] = st.text_input(f"{raw} (free text)", "unknown")
        else:
            input_dict[feature] = st.number_input(feature.replace("_", " "), min_value=0.0)

    if st.button("Predict"):
        try:
            input_dict["assignment__id"] = 0
            spark_df = spark.createDataFrame([Row(**input_dict)])
            prediction_df = model.transform(spark_df)
            predicted_value = prediction_df.select("prediction").collect()[0][0]
            st.success(f"✅ Predicted Milk Quantity Sold: **{predicted_value:.2f} liters**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")

# -------------------- Explore Mode --------------------
elif mode == "Explore Data":
    st.header("Livestock & Milk Data Visualizer")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    vis_type = st.sidebar.selectbox("Choose a visualization", [
        "Milk Production vs Sale",
        "Livestock Deaths (Total)",
        "Livestock Deaths Heatmap",
        "AI Services vs Milk Production",
        "Year-wise Milk Trend",
        "Top 15 Feature Importances"
    ])

    if vis_type == "Milk Production vs Sale":
        fig = px.scatter(df, x="qty_prod_milk", y="qty_sold_milk", color="Dzongkhag",
                         size="qty_consumed_milk", hover_data=["year"],
                         title="Milk Produced vs Sold by Dzongkhag")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "Livestock Deaths (Total)":
        dzongkhags = df["Dzongkhag"].dropna().unique().tolist()
        selected_dzongkhag = st.selectbox("Select Dzongkhag", ["All"] + sorted(dzongkhags))

        death_cols = [col for col in df.columns if col.startswith("No_death_")]
        melted = df[["year", "Dzongkhag"] + death_cols].melt(
            id_vars=["year", "Dzongkhag"], var_name="Animal", value_name="Deaths")
        melted["Animal"] = melted["Animal"].str.replace("No_death_", "").str.replace("_", " ")

        if selected_dzongkhag != "All":
            melted = melted[melted["Dzongkhag"] == selected_dzongkhag]

        summary = melted.groupby(["year", "Animal"])["Deaths"].sum().reset_index()

        fig = px.bar(summary, x="year", y="Deaths", color="Animal", barmode="group",
                     title=f"Year-wise Livestock Deaths" + (f" in {selected_dzongkhag}" if selected_dzongkhag != "All" else ""))
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "Livestock Deaths Heatmap":
        death_cols = [col for col in df.columns if col.startswith("No_death_")]
        melted = df.groupby("Dzongkhag")[death_cols].sum().reset_index().melt(
            id_vars="Dzongkhag", var_name="Livestock Type", value_name="Deaths")
        fig = px.density_heatmap(melted, x="Livestock Type", y="Dzongkhag", z="Deaths",
                                 color_continuous_scale="Reds", title="Deaths by Type and Dzongkhag")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "AI Services vs Milk Production":
        ai_grouped = df.groupby("Did you or your household avail the AI services for your cattles?")["qty_prod_milk"].mean().reset_index()
        fig = px.bar(ai_grouped,
                     x="Did you or your household avail the AI services for your cattles?",
                     y="qty_prod_milk", color="qty_prod_milk",
                     title="AI Service Usage vs Avg Milk Production")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "Year-wise Milk Trend":
        year_milk = df.groupby("year")[["qty_prod_milk", "qty_sold_milk", "qty_consumed_milk"]].sum().reset_index()
        fig = px.line(year_milk, x="year", y=["qty_prod_milk", "qty_sold_milk", "qty_consumed_milk"],
                      markers=True, title="Milk Production, Sale & Consumption Over Years")
        st.plotly_chart(fig, use_container_width=True)

    elif vis_type == "Top 15 Feature Importances":
        try:
            rf_model = model.stages[-1]
            assembler = model.stages[-2]
            importances = rf_model.featureImportances
            input_cols = assembler.getInputCols()

            importance_data = [
                (input_cols[i], importance)
                for i, importance in zip(importances.indices, importances.values)
                if i < len(input_cols)
            ]

            importance_df = pd.DataFrame(importance_data, columns=["Feature", "Importance"])
            importance_df = importance_df.sort_values(by="Importance", ascending=False).head(15)

            st.subheader("Top 15 Features Impacting Milk Sales")
            fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                         title="Top Influential Features")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Unable to load feature importances: {str(e)}")
