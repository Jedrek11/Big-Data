# Databricks notebook source
# MAGIC %md
# MAGIC ## 1.Napisz kod, który ładuje zbiór danych do struktury dataframe i do RDD (movies, users)

# COMMAND ----------

spark

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt


base_path = "/Volumes/workspace/default/lab2/"
path_movies = base_path + "movies.csv"
path_ratings = base_path + "ratings.csv"
path_tags = base_path + "tags.csv"
path_users = base_path + "users.csv"

spark = SparkSession.builder.appName("Lab2_Final").getOrCreate()



# COMMAND ----------

df_movies = spark.read.csv(path_movies, header=True, inferSchema=True)
df_movies.show(3)


print("RDD: Operacja pominięta (nieobsługiwana na klastrze Serverless/Unity Catalog).")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Wyjaśnij czym się różni dataframe od RDD

# COMMAND ----------

# MAGIC %md
# MAGIC ## RDD (Resilient Distributed Dataset)
# MAGIC - to najniższy poziom abstrakcji w Spark,
# MAGIC - przechowuje kolekcję obiektów bez schematu,
# MAGIC - użytkownik sam odpowiada za typy danych i strukturę,
# MAGIC - operacje są niskopoziomowe (map, filter, reduce),
# MAGIC - mało optymalny — Spark nie może analizować planu wykonania,
# MAGIC - trudniejszy do użycia, wymaga więcej kodu i większej ostrożności.
# MAGIC
# MAGIC ## DataFrame
# MAGIC - struktura tabelaryczna (kolumny + typy danych),
# MAGIC - Spark posiada schemat, więc może optymalizować zapytania (Catalyst optimizer), 
# MAGIC - operacje są deklaratywne – podobne do SQL,
# MAGIC - działa szybciej dzięki optymalizacji i wewnętrznemu formatowi Tungsten,
# MAGIC - obsługuje automatyczne konwersje typów, joiny, agregacje itd.,
# MAGIC - znacznie wygodniejszy w analizie danych i machine learningu.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wpróbuj teraz stworzyć schemat typów danych zanim załadujesz dane (movies, users) aby od razu z góry określić typ tych danych

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType


# COMMAND ----------

df_movies.printSchema()
df_movies.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC **4. Dla pliku movies stwórz osobne kolumny dla poszczególnych elementów dat (miesiąc, dzień, godzina)**

# COMMAND ----------

from pyspark.sql.functions import lit, rand, floor

df_movies = df_movies \
    .withColumn("rok", lit(2024)) \
    .withColumn("miesiac", floor(rand()*12 + 1)) \
    .withColumn("dzien", floor(rand()*28 + 1)) \
    .withColumn("godzina", floor(rand()*24))


# COMMAND ----------

df_movies.show(5)


# COMMAND ----------

# MAGIC %md 5. Dla pliku movies rozbij gatunki na osobne wiersze, czyli powtarzasz ten sam film z jednym przypisanym gatunkiem filmy (np. adventure) w jednym wierszu (polecenie explode)
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import split, explode

df_movies_exploded = df_movies \
    .withColumn("genre", explode(split("genres", "\\|"))) \
    .select("movieId", "genre", "title", "rok")


# COMMAND ----------

df_movies_exploded.show(20, truncate=False)


# COMMAND ----------

# MAGIC %md 6. załaduj dane z pliku ratings i wyświetl schemat
# MAGIC

# COMMAND ----------

df_ratings = spark.read.csv(
    "/Volumes/workspace/default/lab2/ratings.csv",
    header=True,
    inferSchema=True
)

df_ratings.printSchema()
df_ratings.show(5)


# COMMAND ----------

# MAGIC %md 7. dodaj kolumny „year, month, day” I wypełnij danymi z timestamp
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import from_unixtime, year, month, dayofmonth

df_ratings = df_ratings.withColumn(
    "datetime", from_unixtime("timestamp")
).withColumn(
    "year", year("datetime")
).withColumn(
    "month", month("datetime")
).withColumn(
    "day", dayofmonth("datetime")
)


# COMMAND ----------

df_ratings.show(10, truncate=False)


# COMMAND ----------

# MAGIC %md 8. oblicz ile było ocen filmów w kolejnych latach
# MAGIC

# COMMAND ----------


df_ratings.groupBy("year").count().orderBy("year").show()

# COMMAND ----------

# MAGIC %md 9. oblicz ile było ocen w kolejnych miesiącach
# MAGIC

# COMMAND ----------


df_ratings.groupBy("month").count().orderBy("month").show()

# COMMAND ----------

# MAGIC %md
# MAGIC 10. załaduj plik tags , wyświetl informacje o zawartości i schemacie danych

# COMMAND ----------


df_tags = spark.read.csv(path_tags, header=True, inferSchema=True)
df_tags.show(3)

# COMMAND ----------

# MAGIC %md 11. przekonwertuj zawartość kolumny z time stamp na dzień, miesiąc i rok
# MAGIC

# COMMAND ----------


df_tags = df_tags.withColumn("datetime", col("timestamp").cast("timestamp")) \
    .withColumn("year", year("datetime")) \
    .withColumn("month", month("datetime")) \
    .withColumn("day", dayofmonth("datetime"))
df_tags.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC 12. podaj liczbę tagów w kolejnych miesiącach i zrób wykres

# COMMAND ----------


tags_per_month = df_tags.groupBy("year", "month").count().orderBy("year", "month").toPandas()

plt.figure(figsize=(10, 5))
plt.bar(tags_per_month['month'], tags_per_month['count'])
plt.xlabel('Miesiąc')
plt.ylabel('Liczba tagów')
plt.title('Liczba tagów w miesiącach')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 13. załaduj plik movies

# COMMAND ----------

from pyspark.sql.functions import avg

df_avg_ratings = df_ratings.groupBy("movieId") \
    .agg(avg("rating").alias("srednia_ocena")) \
    .orderBy("movieId")

df_avg_ratings.show(20)


# COMMAND ----------

# MAGIC %md
# MAGIC 14. wykonaj operację join po polu „moviefield” i dodaj kolumnę z rokiem produkcji, załaduj dane z raings.csv do zmiennej df_ratings
# MAGIC Wykonaj operację join
# MAGIC Np. 
# MAGIC var df_mar = df_movies.join(df_ratings,df_movies.col("movieId").equalTo(df_ratings.col("movieId")));
# MAGIC Jeżeli nazwy kolumn są identyczne, można użyć: 
# MAGIC var df_mr = df_movies.join(df_ratings,"movieId","inner");
# MAGIC

# COMMAND ----------

df_movies_parsed = df_movies.withColumn("year_produced", regexp_extract(col("title"), r"\((\d{4})\)", 1))
df_mr = df_movies_parsed.join(df_ratings, "movieId", "inner")
print("Join wykonany.")

# COMMAND ----------

# MAGIC %md
# MAGIC 15. Zgrupuj dane po tytule używając funkcji groupBy() i dodając kolumny: 
# MAGIC •	z minimalną oceną - nazwa kolumny min_rating 
# MAGIC •	średnią ocen o nazwie avg_rating
# MAGIC •	maksymalną oceną - nazwa kolumny max_rating
# MAGIC •	liczbą ocen - nazwa kolumny rating_cnt
# MAGIC •	Użyj funkcji agg( min(“rating”).alias(“min_rating”), …kolejna agregacja,…). Funkcja alias() służy do zmiany nazwy 
# MAGIC •	Posortuj po liczbie ocen (malejąco). 
# MAGIC

# COMMAND ----------

df_stats = df_mr.groupBy("title").agg(
    min("rating").alias("min_rating"),
    avg("rating").alias("avg_rating"),
    max("rating").alias("max_rating"),
    count("rating").alias("rating_cnt")
).orderBy(col("rating_cnt").desc())
df_stats.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 16. Wyświetl histogram wartości avg_rating, pobierz dane z kolumny za pomocą funkcji select
# MAGIC var avgRatings = df_mr_t.select("avg_rating").where("rating_cnt>=0").as(Encoders.DOUBLE()).collectAsList();
# MAGIC

# COMMAND ----------

rows = df_stats.select("avg_rating").where("rating_cnt>=0").collect()
avg_ratings_list = [row["avg_rating"] for row in rows]

plt.figure()
plt.hist(avg_ratings_list, bins=20, edgecolor='black')
plt.title("Histogram średnich ocen")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 17. Pobierz wartości z kolumny rating_cnt dla rekordów spełniających avg_rating>=4.5. Warunek przekaż jako parametr funkcji where(). Następnie przekonwertuj na zmienną typu List<Double>. 

# COMMAND ----------

high_rating_rows = df_stats.filter(col("avg_rating") >= 4.5).select("rating_cnt").collect()
high_rating_counts = [row["rating_cnt"] for row in high_rating_rows]
print(f"Liczba znalezionych rekordów: {len(high_rating_counts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Przez jaki czas od produkcji filmu pojawiały się oceny?

# COMMAND ----------

# MAGIC %md Przez jaki czas od produkcji filmu pojawiały się oceny?
# MAGIC 18. Dodaj kolumnę release_to_rating_year nadając jej wartośc wyrażenia będącego róznicą roku pobranego z kolumny datetime (data i czas publikacji oceny) oraz year (pochodzącej z tytułu filmu). 
# MAGIC
# MAGIC

# COMMAND ----------

df_diff = df_mr.withColumn("release_to_rating_year", 
                           col("year") - when(col("year_produced") == "", lit(None))
                                         .otherwise(col("year_produced")).cast("int"))
df_diff.select("title", "year_produced", "year", "release_to_rating_year").show(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC 19. Pobierz listę wartości i wyświetl histogram.
# MAGIC 20. Przeprowadź downsampling używając funkcji sample(ratio). Dobierz parametr eksperymentalnie. 
# MAGIC

# COMMAND ----------

df_sampled = df_diff.sample(withReplacement=False, fraction=0.1)
rows = df_sampled.select("release_to_rating_year").na.drop().collect()
diff_list = [row["release_to_rating_year"] for row in rows]

plt.figure()
plt.hist(diff_list, bins=50)
plt.title("Histogram różnicy lat")
plt.show()

# COMMAND ----------

# MAGIC %md 21. Zgrupuj dane po kolumnie release_to_rating_year obliczając liczbę wystąpień. Nowa kolumna ze zagregowanymi dnaymi powinna otrzymać nazwę count. Posortuj dane po release_to_rating_year. 
# MAGIC

# COMMAND ----------

df_diff_grouped = df_diff.groupBy("release_to_rating_year").count().orderBy("release_to_rating_year")
df_diff_grouped.show(5)

# COMMAND ----------

# MAGIC %md 22. Popraw wyrażenie regularne służące do wydzielania tytułu i roku produkcji, aby akceptowało dowolną liczbę spacji po nawiasie zamykającym rok. O ile zmniejszy się liczba wartości NULL? 
# MAGIC

# COMMAND ----------

regexp_extract("title", r"\((\d{4})\)", 1)


# COMMAND ----------

regexp_extract("title", r"\((\d{4})\)\s*", 1)



# COMMAND ----------

from pyspark.sql.functions import regexp_extract, col

# rok ze starego regexu
df_old = df_movies.withColumn(
    "rok_old",
    regexp_extract("title", r"\((\d{4})\)", 1)
)

# rok z poprawionego regexu
df_new = df_old.withColumn(
    "rok_new",
    regexp_extract("title", r"\((\d{4})\)\s*", 1)
)

df_new.select("title", "rok_old", "rok_new").show(20, truncate=False)


# COMMAND ----------

null_old = df_old.filter(col("rok_old") == "").count()
null_new = df_new.filter(col("rok_new") == "").count()

print("NULL przed poprawą:", null_old)
print("NULL po poprawie:", null_new)
print("Zmniejszenie NULL:", null_old - null_new)


# COMMAND ----------

# MAGIC %md
# MAGIC 23. Podczas przetwarzania filmów - w przypadku, kiedy brakuje daty w tytule użyj wartości z kolumny title. Służą do tego funkcje 
# MAGIC when(condition, columnToUse).otherwise(anotherColumnToUse) 
# MAGIC ...
# MAGIC .withColumn("title2",
# MAGIC                     when(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1).equalTo("")
# MAGIC                             ,col("title"))
# MAGIC                             .otherwise(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1)))
# MAGIC  
# MAGIC ...
# MAGIC Nie zmieni to wartości NULL, ale zostanie zachowany tytuł… 
# MAGIC

# COMMAND ----------

df_movies_fixed = df_movies_fixed.withColumn("title_final", 
                                             when(col("title_clean") == "", col("title")).otherwise(col("title_clean")))
df_movies_fixed.filter(col("year_fixed") != "").select("title", "year_fixed").show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC 24. Wyświetlenie histogramu 
# MAGIC •	Stwórz obiekt histogram
# MAGIC •	Pobierz listy z wartościami w kolumnach release_to_rating_year i count. 
# MAGIC •	Pomiń wiersze, w których release_to_rating_year ma wartośc NULL. 
# MAGIC •	Wyświetl histogram. 
# MAGIC

# COMMAND ----------

rows_fixed = df_movies_fixed.filter(col("year_fixed") != "").select("year_fixed").collect()
years_list = [int(row["year_fixed"]) for row in rows_fixed]

plt.figure()
plt.hist(years_list, bins=50)
plt.title("Zad 24: Histogram lat produkcji")
plt.show()

# COMMAND ----------

# MAGIC %md 25. Join MoviesRatingsGenres
# MAGIC Jesteśmy zainteresowani informacjami o ocenach dla gatunków filmów. 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 26. Załaduj filmy i rozbij tablice z gatunkami na poszczególne rekordy. Następnie dokonaj złączenia z ocenami. 
# MAGIC

# COMMAND ----------

df_movies_expl = df_movies.withColumn("genre", explode(split(col("genres"), "\\|")))
df_genre_ratings = df_movies_expl.join(df_ratings, "movieId")
df_genre_ratings.select("title", "genre", "rating").show(5)

# COMMAND ----------

# MAGIC %md 27. Zgrupuj dane po kolumnie genre i wylicz minimalne, średnie, maksymalne oceny dla każdego gatunku oraz liczbę tych ocen. 
# MAGIC

# COMMAND ----------

df_genre_stats = df_genre_ratings.groupBy("genre").agg(
    min("rating").alias("min_rating"),
    avg("rating").alias("avg_rating"),
    max("rating").alias("max_rating"),
    count("rating").alias("rating_cnt")
)
df_genre_stats.show()

# COMMAND ----------

# MAGIC %md 28. Wyświetl 3 kategorie o najwyższych średnich ocenach oraz największej liczbie ocen. 
# MAGIC

# COMMAND ----------

print("\n--- Zadanie 28 ---")
print("Top 3 średnia ocena:")
df_genre_stats.orderBy(col("avg_rating").desc()).show(3)
print("Top 3 liczba ocen:")
df_genre_stats.orderBy(col("rating_cnt").desc()).show(3)

# COMMAND ----------

# MAGIC %md 
# MAGIC 29. Przefiltruj zgrupowane dane pozostawiając tylko te, które mają wartości średnich ocen avg_rating większe niż średnia w całym zbiorze ratings.csv. Uwaga średnia będzie inna, jeżeli zostanie obliczone dla df_ratings (same oceny) i df_mr - połączenie movies i ratings po rozbiciu na gatunki. 
# MAGIC 30. W zasadzie wykonanie (29) wymaga dwóch kwerend (wywołania subquery). Można to również zakodować w SQL tworząc widoki zbiorów danych. 
# MAGIC

# COMMAND ----------

df_genre_stats.createOrReplaceTempView("genre_stats")
df_ratings.createOrReplaceTempView("ratings")

query = """
    SELECT genre, avg_rating, rating_cnt
    FROM genre_stats
    WHERE avg_rating > (SELECT AVG(rating) FROM ratings)
    ORDER BY avg_rating DESC
"""
spark.sql(query).show()

# COMMAND ----------

# MAGIC %md 31. Join UsersTags
# MAGIC

# COMMAND ----------

print("--- Zadanie 31 ---")
df_users = spark.read.csv(path_users, header=True, inferSchema=True)
df_users.createOrReplaceTempView("users")
df_tags.createOrReplaceTempView("tags")

# COMMAND ----------

# MAGIC %md 32. Utwórz złączony zbiór za pomocą kwerendy SQL 
# MAGIC

# COMMAND ----------

query_tags = "SELECT u.email, t.tag FROM users u JOIN tags t ON u.userId = t.userId"
df_ut = spark.sql(query_tags)

# COMMAND ----------

# MAGIC %md 33. Zgrupuj dane po kolumnie email 
# MAGIC •	Wzynacz listę tagów podczas grupowania za pomocą funkcji collect_list()
# MAGIC •	Sklej listę tekstów (wprowadzając separator spacji) za pomocą funkcji concat_ws()
# MAGIC
# MAGIC

# COMMAND ----------

print("\n--- Zadanie 33 ---")
df_ut_grouped = df_ut.groupBy("email").agg(
    concat_ws(" ", collect_list("tag")).alias("tags_list")
)
df_ut_grouped.show(5, truncate=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC 34. Join UserRatings Wczytaj dane użytkowników do zbioru df_users oraz oceny do df_ratings. 
# MAGIC 35. Złącz zbiory danych na podstawie identyfikatora użytkownika 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

df_ur = df_users.join(df_ratings, "userId")

# COMMAND ----------

# MAGIC %md 36. Zgrupuj dane po kolumnie email agregując średnie oceny i liczbę ocen. Wyświetl dane, np. posortowane po średniej 
# MAGIC

# COMMAND ----------

df_user_stats = df_ur.groupBy("email").agg(
    avg("rating").alias("avg_rating"),
    count("rating").alias("cnt")
).orderBy(col("avg_rating").desc())
df_user_stats.show(5)

# COMMAND ----------

# MAGIC %md 37. Wyświetl wykres 
# MAGIC Wyświetl wykres punktowy, plt.plot().add(x, y,“o”).label(“data”);, w którym 
# MAGIC •	współrzędna x to avg_rating użytkownika
# MAGIC •	współrzędna y odpowiada kolumnie count
# MAGIC
# MAGIC

# COMMAND ----------

pdf_user_stats = df_user_stats.toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pdf_user_stats['avg_rating'], pdf_user_stats['cnt'], alpha=0.5)
plt.xlabel('Średnia ocena')
plt.ylabel('Liczba oddanych głosów')
plt.title('Aktywność użytkowników vs Oceny')
plt.grid(True)
plt.show()