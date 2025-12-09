# Databricks notebook source
spark

# COMMAND ----------

# # Example: Read a table into a Spark DataFrame and display it
# df = spark.table("my_table")
# display(df)

# COMMAND ----------

df_prac = spark.read.csv(
    "/Volumes/workspace/default/lab1/pracownicy.csv",
    header=True,
    inferSchema=True
)
df_prac.show()


# COMMAND ----------

df_proj = spark.read.csv(
    "/Volumes/workspace/default/lab1/projekty.csv",
    header=True,
    inferSchema=True
)
df_proj.show()


# COMMAND ----------

# 1. Wczytujemy plik jako DataFrame (jedna kolumna "value")
df_text = spark.read.text("/Volumes/workspace/default/lab1/tekst_do_liczenia.txt")
df_text.show(truncate=False)


# COMMAND ----------

df_prac.printSchema()


# COMMAND ----------

df_prac = spark.read.csv(
    "/Volumes/workspace/default/lab1/pracownicy.csv",
    header=True,
    inferSchema=True
)

df_prac = df_prac.toDF("name", "age", "department")

df_prac.show()
df_prac.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2
# MAGIC Z DataFrame wybierz tylko imię i wiek.

# COMMAND ----------

df_selected = df_prac.select("name", "age")
df_selected.show()


# COMMAND ----------

df_prac = spark.read.csv(
    "/Volumes/workspace/default/lab1/pracownicy.csv",
    header=True,
    inferSchema=True
)

df_prac = df_prac.toDF("name", "age", "department")
df_prac.printSchema()


# COMMAND ----------

new_data = [
    ("Jan", 28, "IT"),
    ("Anna", 34, "HR"),
    ("Piotr", 22, "Finance"),
    ("Ewa", 45, "IT"),
    ("Marek", 31, "HR"),
    ("Joanna", 29, "Finance")
]

df_prac = spark.createDataFrame(new_data, ["name", "age", "department"])
df_prac.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 
# MAGIC Przefiltruj dane, wybierając osoby powyżej 30 lat.

# COMMAND ----------

df_prac.filter(df_prac.age > 30).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.	Sortowanie i agregacje

# COMMAND ----------

df_sorted = df_prac.orderBy(df_prac.age.desc())
df_sorted.show()


# COMMAND ----------

from pyspark.sql.functions import avg

df_grouped = df_prac.groupBy("department").agg(avg("age").alias("avg_age"))
df_grouped.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.	Dodanie i usunięcie kolumn

# COMMAND ----------

from pyspark.sql.functions import col

df_new = df_prac.withColumn("age_in_5_years", col("age") + 5)
df_new.show()


# COMMAND ----------

df_removed = df_new.drop("department")
df_removed.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.	Łączenie dwóch DataFrame

# COMMAND ----------

data_projects = [
    ("Jan", 5),
    ("Anna", 3),
    ("Piotr", 4),
    ("Ewa", 6),
    ("Marek", 2),
    ("Joanna", 7)
]

df_proj = spark.createDataFrame(data_projects, ["name", "projects"])
df_proj.show()


# COMMAND ----------

df_joined = df_prac.join(df_proj, on="name", how="inner")
df_joined.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.	Obsługa brakujących wartości
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import when, col

df_null = df_prac.withColumn(
    "age",
    when(col("name").isin("Jan", "Marek"), None).otherwise(col("age"))
)

df_null.show()



# COMMAND ----------

df_filled = df_null.na.fill({"age": 30})
df_filled.show()



# COMMAND ----------

df_dropped = df_null.na.drop()
df_dropped.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.	Operacje na oknach (Window Functions)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col

window_spec = Window.orderBy(col("projects").desc())


# COMMAND ----------

df_ranked = df_joined.withColumn("rank", rank().over(window_spec))
df_ranked.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.	Operacje na RDD – klasyczne liczenie słów

# COMMAND ----------

df_text = spark.read.text("/Volumes/workspace/default/lab1/tekst_do_liczenia.txt")
df_text.show(truncate=False)


# COMMAND ----------

from pyspark.sql.functions import split, explode, lower, col, regexp_replace

df_clean = df_text.select(
    regexp_replace(col("value"), r"[^\p{L}0-9]+", " ").alias("value")
)

df_words = df_clean.select(
    explode(split(col("value"), " ")).alias("word")
)

df_words = df_words.filter(col("word") != "").select(lower(col("word")).alias("word"))

df_word_counts = df_words.groupBy("word").count().orderBy(col("count").desc())

df_word_counts.show(20)




# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.	Zapisywanie wyników do pliku
# MAGIC

# COMMAND ----------

df_joined.write.mode("overwrite").csv("/Volumes/workspace/default/lab1/wynik_lab01_csv", header=True)

# COMMAND ----------

df_joined.write.mode("overwrite").json("/Volumes/workspace/default/lab1/wynik_lab01_json")
