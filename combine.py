from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# STEP 1: Start Spark session
spark = SparkSession.builder.appName("dataset").getOrCreate()

# STEP 2: Load CSVs and add year column
df_2020 = spark.read.csv("/opt/spark/LC 2020.csv", header=True, inferSchema=True).withColumn("year", lit(2020))
df_2021 = spark.read.csv("/opt/spark/LC 2021.csv", header=True, inferSchema=True).withColumn("year", lit(2021))
df_2022 = spark.read.csv("/opt/spark/2022.csv", header=True, inferSchema=True).withColumn("year", lit(2022))
df_2023 = spark.read.csv("/opt/spark/2023.csv", header=True, inferSchema=True).withColumn("year", lit(2023))
df_2024 = spark.read.csv("/opt/spark/2024.csv", header=True, inferSchema=True).withColumn("year", lit(2024))

# STEP 3: Find common columns across all DataFrames
common_cols = list(
    set(df_2020.columns)
    & set(df_2021.columns)
    & set(df_2022.columns)
    & set(df_2023.columns)
    & set(df_2024.columns)
)

# STEP 4: Select common columns and union all
df_all = df_2020.select([col(f"`{c}`") for c in common_cols]) \
    .unionByName(df_2021.select([col(f"`{c}`") for c in common_cols])) \
    .unionByName(df_2022.select([col(f"`{c}`") for c in common_cols])) \
    .unionByName(df_2023.select([col(f"`{c}`") for c in common_cols])) \
    .unionByName(df_2024.select([col(f"`{c}`") for c in common_cols]))

# Show result
df_all.show(1)

# STEP 5: Save the combined DataFrame
df_all.coalesce(1).write.mode("overwrite").option("header", True).csv("/opt/spark/combined_output")
