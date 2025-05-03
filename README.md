# Assignment-5-FakeNews-Detection


---

````markdown
# Fake News Detection with Spark MLlib

This project builds a simple machine learning pipeline using **Apache Spark MLlib** to classify news articles as **FAKE** or **REAL** based on their content.

---

## Objectives

By completing these tasks, you will gain experience with:

- ✅ Loading and exploring a text dataset in Spark.
- ✅ Preprocessing text using `Tokenizer` and `StopWordsRemover`.
- ✅ Extracting features using **TF-IDF** (`HashingTF` and `IDF`).
- ✅ Training a **Logistic Regression** classifier using Spark MLlib.
- ✅ Evaluating a binary classification model using **accuracy** and **F1 score**.

---

##  Dataset Used

- File: `fake_news_sample.csv`
- Contains 500 synthetic news articles (250 FAKE, 250 REAL).
- Each article includes:
  - `id`: Unique identifier
  - `title`: News headline
  - `text`: News article content
  - `label`: Either FAKE or REAL

---

##  Task Breakdown

| Task | Description | Output File |
|------|-------------|-------------|
| **Task 1** | Load data, explore schema, show sample rows, count articles | `task1_output.csv` |
| **Task 2** | Preprocess text: lowercase, tokenize, remove stopwords | `task2_output.csv` |
| **Task 3** | Extract features using TF-IDF, convert labels to numeric | `task3_output.csv` |
| **Task 4** | Train Logistic Regression model, generate predictions | `task4_output.csv` |
| **Task 5** | Evaluate model using accuracy and F1 score | `task5_output.csv` |

---

##  How to Run the Code

Run the entire pipeline using:

```bash
python3 fake_news.py
````

>  `spark-submit` is not required for this assignment, as the code uses Spark locally via `SparkSession`.

---

##  Requirements

* Python 3.x
* `pyspark`
* `pandas`
* `faker` (used only for dataset generation)

Install dependencies (if needed):

```bash
pip install pyspark pandas faker
```

---

## Notes

* The script introduces 5% random label noise to make classification more realistic.
* All intermediate results are saved as CSVs for inspection and evaluation.
* The project demonstrates a complete ML pipeline using PySpark and Spark MLlib.


