# Forest Cover Type Prediction Using Classical Machine Learning

This project involved applying **classical machine learning techniques** to the **Forest Covertypes dataset**, a real-world dataset for predicting **forest cover types based on cartographic features**. The goal was to **develop clustering and classification models**, evaluate their performance and derive insights using **internal and external evaluation metrics**.

## Key Contributions & Achievements

### 📌 Clustering Analysis (Unsupervised Learning)
- Implemented **seven clustering algorithms**:
  - **Hierarchical Clustering** (Best performer)
  - **Agglomerative Clustering**
  - **K-Means Clustering**
  - **Gaussian Mixture Model (GMM)**
  - **Density-Based Spatial Clustering (DBSCAN)**
  - **BIRCH Clustering**
  - **Mean Shift Clustering**
- **Performance Evaluation:**
  - **Internal Metrics:** Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
  - **External Metrics:** Adjusted Rand Index, Adjusted Mutual Information
- **Key Findings:**
  - **Hierarchical Clustering** was the best-performing method showing high agreement with actual groups.
  - **Mean Shift and BIRCH struggled** due to poor cluster separation.
  - **K-Means and Agglomerative performed well** demonstrating strong cluster compactness.

### 📌 Classification Analysis (Supervised Learning)
- Built **seven classification models**:
  - **Extra Trees Classifier** (Best performer)
  - **Random Forest**
  - **Decision Trees**
  - **AdaBoost**
  - **K-Nearest Neighbours (KNN)**
  - **Logistic Regression**
  - **Gaussian Naïve Bayes**
- **Performance Evaluation:**
  - **Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC**
  - **Confusion Matrices & ROC Curves** for model comparison.
- **Key Findings:**
  - **Extra Trees Classifier was the best model**, achieving the highest scores across all evaluation metrics.
  - **Random Forest had high precision and recall**, but struggled with overall accuracy.
  - **KNN and Logistic Regression underperformed**, indicating challenges in classifying instances correctly.
  - **AdaBoost showed moderate performance**, excelling in ROC-AUC but having low balanced accuracy.

### 📌 Model Optimization & Feature Engineering
- **Hyperparameter tuning** (e.g., adjusting cluster sizes, tree depths, boosting rounds).
- **Dimensionality reduction** using **PCA** for improved classification efficiency.
- **Feature scaling** (**MinMaxScaler, StandardScaler**) to normalise input data.

### 📌 Key Findings & Business Impact
- **Hierarchical Clustering is recommended** for segmenting **forest cover types** due to its superior performance in grouping.
- **Extra Trees Classifier is the optimal classification model**, offering **high accuracy and robustness**.
- **Feature engineering & dimensionality reduction** significantly improved **model efficiency**.

---

## 🛠 Technologies & Tools Used
- **Python** (Scikit-Learn, NumPy, Pandas, Matplotlib, Seaborn)
- **Unsupervised Learning** (Clustering: K-Means, DBSCAN, Hierarchical)
- **Supervised Learning** (Classification: Extra Trees, Decision Trees, Random Forest, AdaBoost)
- **Model Evaluation** (Confusion Matrices, ROC Curves, Precision-Recall)
- **Dimensionality Reduction** (PCA) & **Feature Engineering**

🚀 **This project demonstrated my ability to apply machine learning algorithms, evaluate model performance and optimise predictive models for real-world datasets.**
