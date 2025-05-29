from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, round as round_
from pyspark.sql.types import StringType, NumericType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ------------------------ 1. Start Spark session ------------------------

# spark = SparkSession.builder \
#     .appName("Milk Sales Prediction") \
#     .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
#     .getOrCreate()
spark = SparkSession.builder.appName("Livestock Datasets").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# ------------------------ 2. Load dataset from HDFS ------------------------
# df = spark.read.csv("hdfs://namenode:8020/Data/combine_data.csv", header=True, inferSchema=True)
df = spark.read.csv("/opt/spark/combine_data.csv", header=True, inferSchema=True)


df = df.toDF(*[c.replace('.', '_') for c in df.columns])

df.show(1)
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
df.printSchema()

# ------------------------ 4. Null & column type analysis ------------------------
categorical_cols, numerical_cols = [], []
for field in df.schema.fields:
    if isinstance(field.dataType, StringType):
        categorical_cols.append(field.name)
    elif isinstance(field.dataType, NumericType):
        numerical_cols.append(field.name)

print("\nNull counts per column:")
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull() | (col(col_name) == "") | (col(col_name) == "null")).count()
    print(f"{col_name}: {null_count}")

# ------------------------ 5. Clean categorical columns ------------------------
for col_name in categorical_cols:
    df = df.withColumn(col_name, when(col(col_name).isNull() | (col(col_name) == "") | (col(col_name) == "null"), "unknown").otherwise(col(col_name)))
    df = df.withColumn(col_name, when(col(col_name).rlike("^[0-9.]+$"), "unknown").otherwise(col(col_name)))

# Specific corrections 
df = df.withColumn("Dzongkhag", when(col("Dzongkhag") == "Monggar", "Mongar")
    .when(col("Dzongkhag") == "Pema Gatshel", "Pemagatshel")
    .when(col("Dzongkhag") == "Trashi Yangtse", "Trashiyangtse")
    .when(col("Dzongkhag") == "Lhuentse", "Lhuntse")
    .otherwise(col("Dzongkhag")))

if "Area" in df.columns:
    df = df.withColumn("Area", when(col("Area").isin(["Rural", "Urban", "unknown"]), col("Area")).otherwise("unknown"))

# ------------------------  6. Fill numeric nulls with mean ------------------------
means_dict = df.select([mean(c).alias(c) for c in numerical_cols]).collect()[0].asDict()
df = df.na.fill(means_dict)
# ------------------------  7. Round numeric values ------------------------
for col_name in numerical_cols:
    df = df.withColumn(col_name, round_(col(col_name), 2))

# ------------------------  8. Outlier detection & Winsorization ------------------------
def winsorize_iqr(df, num_cols, exclude_cols=["year"]):
    for col_name in num_cols:
        if col_name in exclude_cols:
            continue
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        if len(quantiles) < 2: continue
        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df.withColumn(col_name, when(col(col_name) < lower, lower)
            .when(col(col_name) > upper, upper)
            .otherwise(col(col_name)))
    return df

df = winsorize_iqr(df, numerical_cols)

#Recheck outliers after treatment
print("\nRechecking Outliers After Winsorization:")
for col_name in numerical_cols:
    if col_name == "year": continue
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    if len(quantiles) < 2: continue
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count_outliers = df.filter((col(col_name) < lower) | (col(col_name) > upper)).count()
    print(f"{col_name}: Outliers Remaining = {count_outliers}")

# ------------------------ 9. Feature Engineering ------------------------
indexers = [StringIndexer(inputCol=c, outputCol=c + "_indexed", handleInvalid="keep") for c in categorical_cols]
encoded_cols = [c + "_indexed" for c in categorical_cols]
encoder = OneHotEncoder(inputCols=encoded_cols, outputCols=[c.replace("_indexed", "_vec") for c in encoded_cols])
onehot_output_cols = [c.replace("_indexed", "_vec") for c in encoded_cols]
feature_cols = [col for col in numerical_cols if col not in ["qty_sold_milk", "year"]] + onehot_output_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ------------------------ 10. Model Definition ------------------------
gbt = GBTRegressor(featuresCol="features", labelCol="qty_sold_milk", predictionCol="prediction")

pipeline = Pipeline(stages=indexers + [encoder, assembler, gbt])

# ------------------------ 11. Model Training ------------------------
pipeline_model = pipeline.fit(df)
predictions = pipeline_model.transform(df)

# ------------------------ 12. Evaluation ------------------------
evaluator_rmse = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="mae")
evaluator_mse = RegressionEvaluator(labelCol="qty_sold_milk", predictionCol="prediction", metricName="mse")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
mse = evaluator_mse.evaluate(predictions)

print(f"\nGBT Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# ------------------------ 13. Show Predictions ------------------------
predictions.select("Dzongkhag", "year", "qty_sold_milk", "prediction").orderBy("Dzongkhag", "year").show(20, truncate=False)

# ------------------------ 14. Save Model (Optional) ------------------------
# pipeline_model.write().overwrite().save("hdfs://namenode:8020/Data/models/gbt_milk_model")
pipeline_model.write().overwrite().save("/opt/spark/models/rf_milk_model")


