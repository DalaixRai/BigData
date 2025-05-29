import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

st.set_page_config(page_title="Milk Sales Predictor", layout="centered")

# ------------------------ 🔹 Spark Setup ------------------------
@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("StreamlitRFApp").getOrCreate()

spark = get_spark_session()

# ------------------------ 🔹 Load Model ------------------------
@st.cache_resource
def load_model():
    return PipelineModel.load("/opt/spark/models/rf_milk_model")

model = load_model()

# ------------------------ 🔹 Streamlit UI ------------------------
st.title("🍼 Milk Sales Predictor")
st.markdown("Enter livestock and production details to predict **qty_sold_milk**.")
st.header("📥 Enter Input Data")

# Manually handled numeric features
input_dict = {
    "qty_prod_milk": st.number_input("Qty Produced Milk", min_value=0.0),
    "qty_consumed_milk": st.number_input("Qty Consumed Milk", min_value=0.0),
    "qty_processed_milk": st.number_input("Qty Processed Milk", min_value=0.0),
    "qty_sold_butter": st.number_input("Qty Sold Butter", min_value=0.0),
    "qty_prod_Chicken": st.number_input("Qty Produced Chicken", min_value=0.0),
    "qty_prod_Fish": st.number_input("Qty Produced Fish", min_value=0.0),
    "qty_prod_Yakmeat": st.number_input("Qty Produced Yak Meat", min_value=0.0),
}

# ------------------------ 🔹 Feature Handling ------------------------

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

# Define known categorical options
categorical_options = {
    "Dzongkhag": [
        "Bumthang", "Chhukha", "Dagana", "Gasa", "Haa", "Lhuntse", "Mongar", "Paro", "Pemagatshel", "Punakha",
        "Samdrup Jongkhar", "Samtse", "Sarpang", "Thimphu", "Trashigang", "Trashiyangtse", "Trongsa",
        "Tsirang", "Wangdue Phodrang", "Zhemgang", "unknown"
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

# Generate inputs
for feature in auto_features:
    if feature.endswith("_vec"):
        raw_col = feature.replace("_vec", "")
        if raw_col in categorical_options:
            input_dict[raw_col] = st.selectbox(raw_col, categorical_options[raw_col])
        else:
            input_dict[raw_col] = st.text_input(f"{raw_col} (free text)", "unknown")
    else:
        input_dict[feature] = 0.0

# ------------------------ 🔹 Predict ------------------------

if st.button("🔮 Predict Milk Quantity Sold"):
    try:
        input_dict["assignment__id"] = 0  # Auto-added backend feature
        spark_df = spark.createDataFrame([Row(**input_dict)])
        prediction_df = model.transform(spark_df)
        predicted_value = prediction_df.select("prediction").collect()[0][0]
        st.success(f"✅ Predicted Milk Quantity Sold: **{predicted_value:.2f} liters**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
