from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean
from pyspark.sql.types import StringType, NumericType
from pyspark.sql import DataFrame  
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import round as round_
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col


# ------------------------ 1. Start Spark session ------------------------
spark = SparkSession.builder.appName("Livestock Datasets").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ------------------------ 2. Load and clean dataset ------------------------
df = spark.read.csv("/opt/spark/combine_data.csv", header=True, inferSchema=True)
df = df.toDF(*[c.replace('.', '_') for c in df.columns])  # Replace dots in column names

# ------------------------ 3. Dataset overview ------------------------
df.show(1)
num_rows = df.count()
num_cols = len(df.columns)

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")
df.printSchema()

# ------------------------ 4. Null value report ------------------------
print("\n🔍 Null counts for each column:")
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull() | (col(col_name) == "") | (col(col_name) == "null")).count()
    print(f"{col_name}: {null_count} nulls")

print("\nColumns with 100% null or empty values:")
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull() | (col(col_name) == "") | (col(col_name) == "null")).count()
    if null_count == num_rows:
        print(f"{col_name}: 100% null ({null_count}/{num_rows})")

# ------------------------ 5. Split categorical and numerical columns ------------------------
categorical_cols = []
numerical_cols = []

for field in df.schema.fields:
    if isinstance(field.dataType, StringType):
        categorical_cols.append(field.name)
    elif isinstance(field.dataType, NumericType):
        numerical_cols.append(field.name)

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print(f"Number of categorical columns: {len(categorical_cols)}")
print(f"Number of numerical columns: {len(numerical_cols)}")

# ------------------------ 6. Replace nulls in categorical columns with 'unknown' ------------------------

def fill_categorical_nulls(df: DataFrame, cat_cols: list) -> DataFrame:
    for col_name in cat_cols:
        df = df.withColumn(
            col_name,
            when(
                col(col_name).isNull() | (col(col_name) == "") | (col(col_name) == "null"),
                "unknown"
            ).otherwise(col(col_name))
        )
    return df

def replace_numeric_strings_in_categoricals(df: DataFrame, cat_cols: list) -> DataFrame:
    for col_name in cat_cols:
        df = df.withColumn(
            col_name,
            when(col(col_name).rlike("^[0-9.]+$"), "unknown").otherwise(col(col_name))
        )
    return df


# Apply the transformation
# Step 1: Fill nulls/empty/"null" with "unknown"
df_cleaned = fill_categorical_nulls(df, categorical_cols)

# Step 2: Replace numeric strings in categorical columns with "unknown"
df_cleaned = replace_numeric_strings_in_categoricals(df_cleaned, categorical_cols)

df_cleaned.select("Dzongkhag").distinct().show(truncate=False)

df_cleaned = df_cleaned.withColumn(
    "Dzongkhag",
    when(col("Dzongkhag") == "Monggar", "Mongar")
    .when(col("Dzongkhag") == "Pema Gatshel", "Pemagatshel")
    .when(col("Dzongkhag") == "Trashi Yangtse", "Trashiyangtse")
    .when(col("Dzongkhag") == "Lhuentse", "Lhuntse")
    .otherwise(col("Dzongkhag"))
)

# Step 3b: Clean invalid Area values (like 'Yes')
df_cleaned = df_cleaned.withColumn(
    "Area",
    when(col("Area").isin(["Rural", "Urban", "unknown"]), col("Area")).otherwise("unknown")
)


# Show only the cleaned categorical columns
print("Columns filled with 'unknown':", categorical_cols)
df_cleaned.select(categorical_cols).show(10, truncate=False)


# ------------------------ 7. Fill nulls in numerical columns with mean ------------------------

# Compute column-wise means
means_dict = df_cleaned.select([mean(c).alias(c) for c in numerical_cols]).collect()[0].asDict()

# Fill numerical nulls with column mean
df_filled = df_cleaned.na.fill(means_dict)

# Round all numeric columns (optional)
for col_name in numerical_cols:
    df_filled = df_filled.withColumn(col_name, round_(col(col_name), 2))

# ------------------------ 8. Export combined cleaned data (categorical + numerical) ------------------------

df_filled.write.mode("overwrite").option("header", True).csv("/opt/spark/data/cleaned_combined_data")


# Step 1: Index categorical columns
indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
    for col_name in categorical_cols
]

# Step 2: OneHotEncode the indexed columns
encoded_cols = [col + "_indexed" for col in categorical_cols]
encoder = OneHotEncoder(
    inputCols=encoded_cols,
    outputCols=[col.replace("_indexed", "_vec") for col in encoded_cols],
    handleInvalid="keep"
)

# Step 3: Build and apply pipeline
pipeline = Pipeline(stages=indexers + [encoder])
df_final = pipeline.fit(df_cleaned).transform(df_cleaned)

# ------------------------  6. Show one-hot encoded columns ------------------------
one_hot_cols = [col.replace("_indexed", "_vec") for col in encoded_cols]
df_final.select(one_hot_cols).show(5, truncate=False)

# After fitting your pipeline
pipeline_model = pipeline.fit(df_cleaned)

# View labels for a specific column
indexer_stage = pipeline_model.stages[categorical_cols.index("Area")]
print("Labels used by StringIndexer:", indexer_stage.labels)


# Function to calculate IQR and detect outliers for each numerical column
def detect_outliers_iqr(df: DataFrame, num_cols: list):
    outlier_summary = []

    for col_name in num_cols:
        # Calculate Q1 and Q3
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        if len(quantiles) < 2:
            continue  # Skip if quantiles can't be computed

        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound))
        count_outliers = outliers.count()

        outlier_summary.append((col_name, Q1, Q3, lower_bound, upper_bound, count_outliers))
    
    return outlier_summary


# Run outlier detection
outlier_info = detect_outliers_iqr(df_cleaned, numerical_cols)

# Display results
print("📌 Outlier Summary (IQR Method):")
for col_name, Q1, Q3, lower, upper, count in outlier_info:
    print(f"{col_name}: Q1={Q1}, Q3={Q3}, Lower={lower}, Upper={upper}, Outliers={count}")


def remove_outliers_iqr(df: DataFrame, num_cols: list) -> DataFrame:
    for col_name in num_cols:
        # Compute Q1 and Q3 for each column
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        if len(quantiles) < 2:
            continue  # Skip if quantiles cannot be computed

        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out rows where values fall outside the bounds
        df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

    return df




def winsorize_iqr(df: DataFrame, num_cols: list, exclude_cols: list = ["year"]) -> DataFrame:
    """
    Applies IQR-based Winsorization by capping outliers and overwriting original columns.
    """
    for col_name in num_cols:
        if col_name in exclude_cols:
            continue
        
        # Calculate Q1 and Q3
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        if len(quantiles) < 2:
            continue
        
        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Overwrite the original column with winsorized values
        df = df.withColumn(
            col_name,
            when(col(col_name) < lower, lower)
            .when(col(col_name) > upper, upper)
            .otherwise(col(col_name))
        )
        
    return df

# 1. Apply winsorization
df_winsorized = winsorize_iqr(df_filled, numerical_cols)

# 2. View results
winsorized_cols = [col_name for col_name in numerical_cols if col_name != "year"]
df_winsorized.select(winsorized_cols).show(10, truncate=False)
df_winsorized.write.mode("overwrite").option("header", True).csv("/opt/spark/export/winsorized_data")


def detect_outliers_post_winsorization(df: DataFrame, num_cols: list, exclude_cols: list = ["year"]):
    print("📌 Rechecking Outliers After Winsorization:\n")
    for col_name in num_cols:
        if col_name in exclude_cols:
            continue
        
        # Calculate Q1 and Q3
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        if len(quantiles) < 2:
            continue

        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count remaining outliers
        count_outliers = df.filter((col(col_name) < lower) | (col(col_name) > upper)).count()

        print(f"{col_name}: Q1={Q1}, Q3={Q3}, Lower={lower}, Upper={upper}, Remaining Outliers={count_outliers}")

detect_outliers_post_winsorization(df_winsorized, numerical_cols)


# 1. Index categorical columns
indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
    for col_name in categorical_cols
]

# 2. One-hot encode indexed columns
encoded_cols = [col + "_indexed" for col in categorical_cols]
encoder = OneHotEncoder(
    inputCols=encoded_cols,
    outputCols=[col.replace("_indexed", "_vec") for col in encoded_cols],
    handleInvalid="keep"
)
onehot_output_cols = [col.replace("_indexed", "_vec") for col in encoded_cols]

# 3. Assemble features
feature_cols = [col for col in numerical_cols if col not in ["qty_sold_milk", "year"]] + onehot_output_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 4. RandomForest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="qty_sold_milk", predictionCol="prediction")

# 5. Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, rf])

# 6. Fit pipeline
pipeline_model = pipeline.fit(df_winsorized)

# 7. Apply model
predictions = pipeline_model.transform(df_winsorized)

# 8. Evaluate
evaluator_rmse = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 9. View predictions
predictions.select("Dzongkhag", "year", "qty_sold_milk", "prediction").orderBy("Dzongkhag", "year").show(20, truncate=False)

evaluator_mae = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="mae")
evaluator_mse = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="mse")

mae = evaluator_mae.evaluate(predictions)
mse = evaluator_mse.evaluate(predictions)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

assembler_stage = pipeline_model.stages[-2]  # VectorAssembler is before the final regressor

final_features = assembler_stage.getInputCols()
print("📌 Final features used for prediction:")
for f in final_features:
    print(f"• {f}")

# Save the trained model
# pipeline_model.write().overwrite().save("/opt/spark/models/rf_milk_model")

