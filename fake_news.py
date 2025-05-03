from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col, when, rand
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("FakeNewsDetectionPipeline").getOrCreate()

# ----------------------
# Task 1: Load & Explore
# ----------------------
print("\n=== Task 1: Load & Basic Exploration ===")
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Add 5% noise to labels by flipping them randomly
df = df.withColumn(
    "label",
    when(rand() < 0.05, when(col("label") == "FAKE", "REAL").otherwise("FAKE")).otherwise(col("label"))
)

df.createOrReplaceTempView("news_data")

df.show(5)
print("Total Articles:", df.count())
df.select("label").distinct().show()

# Save output
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# -----------------------
# Task 2: Preprocess Text
# -----------------------
print("\n=== Task 2: Text Preprocessing ===")
df_clean = df.withColumn("text_lower", lower(col("text")))
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
df_tokenized = tokenizer.transform(df_clean)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)
df_token_output = df_filtered.select("id", "title", "filtered_words", "label")
df_token_output.toPandas().to_csv("task2_output.csv", index=False)

# --------------------------
# Task 3: Feature Extraction
# --------------------------
print("\n=== Task 3: Feature Extraction ===")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df_filtered)
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

df_final.select("id", "filtered_words", "features", "label_index") \
    .toPandas().to_csv("task3_output.csv", index=False)

# -----------------------
# Task 4: Model Training
# -----------------------
print("\n=== Task 4: Model Training ===")
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=30)

lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

predictions = model.transform(test_data)
predictions.select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("task4_output.csv", index=False)

# -----------------------
# Task 5: Evaluation
# -----------------------
print("\n=== Task 5: Model Evaluation ===")
evaluator1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator2 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator1.evaluate(predictions)
f1_score = evaluator2.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Save evaluation
pd.DataFrame([
    {"Metric": "Accuracy", "Value": accuracy},
    {"Metric": "F1 Score", "Value": f1_score}
]).to_csv("task5_output.csv", index=False)

print("\nAll tasks completed. CSVs saved for each step.")
