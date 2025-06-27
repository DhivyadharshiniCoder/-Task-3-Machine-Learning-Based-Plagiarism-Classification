# 🔍 Task 3 – Machine Learning-Based Plagiarism Classification

This project continues from Task 2 and builds a machine learning classification model to automatically detect plagiarism between sentence pairs using their semantic similarity scores.

---

## 📁 Dataset

- `similarity_output_task3.csv` contains:
  - `Sentence_File1`: A sentence taken from the first input document
  - `Sentence_File2`: A sentence taken from the second input document
  - `Similarity_Score`: Cosine similarity between the two sentences (range: 0 to 1)
  - `Plagiarised`: Binary label → `1` if similarity ≥ 0.6 (plagiarised), else `0`

---

## 🎯 Objective

Train a classification model using similarity scores to determine whether sentence pairs are plagiarised.

---

## ⚙️ Steps Performed

1. Loaded the similarity dataset from CSV  
2. Created a binary `Plagiarised` label:
   - 1 if `Similarity_Score ≥ 0.6`
   - 0 otherwise  
3. Split the dataset into train and test sets (80-20 split with stratification)
4. Trained a Logistic Regression model
5. Evaluated model performance using:
   - Accuracy
   - F1 Score
   - Precision and Recall
   - Confusion Matrix

---

## 🤖 Model Used

### 🔹 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

🖼️ Output Screenshot
task3_output.png

