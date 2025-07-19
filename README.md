<div align="center">

# Celebal Customer Churn Prediction Project 

### An End-to-End Machine Learning Project with a Deployed Streamlit Web Application

<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/24464af2-9968-4225-ae07-64b55511c752" />


</div>

---

## 🚀 Live Application

Experience the live prediction model here:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([[YOUR_STREAMLIT_APP_LINK](https://celebal-customer-churn-prediction-project-arrugyh5kzfzapfhcxqk.streamlit.app/)])

---

## 🎯 Project Goal

The primary objective of this project is to develop a robust machine learning model that can accurately predict customer churn for a telecommunications company. By identifying customers who are likely to leave, the company can proactively take measures (like offering discounts or better services) to retain them, thereby reducing revenue loss and improving customer satisfaction.

---

## 🛠️ Technology Stack

This project leverages a variety of modern data science and web development tools:

| Technology | Description |
| :--- | :--- |
| **Python** | Core language for data analysis and model building. |
| **Pandas & NumPy** | For efficient data manipulation and numerical operations. |
| **Matplotlib & Seaborn** | For comprehensive data visualization and EDA. |
| **Scikit-learn** | For building and evaluating machine learning models. |
| **XGBoost** | For training the final, high-performance gradient boosting model. |
| **Imbalanced-learn** | For handling class imbalance using the SMOTE technique. |
| **Streamlit** | For creating and deploying the interactive web application. |
| **Jupyter Notebook** | For the initial research, analysis, and model experimentation. |
| **Joblib** | For saving and loading the trained model and scaler. |

---

##  Workflow

The project follows a structured end-to-end machine learning pipeline:

1.  **Data Cleaning & Preprocessing:** Handled missing values, corrected data types, and applied log transformation to skewed features like `TotalCharges`.
2.  **Exploratory Data Analysis (EDA):** Performed univariate and bivariate analysis to understand feature distributions and their relationship with the target variable (Churn).
3.  **Feature Engineering:** Encoded categorical variables using one-hot encoding.
4.  **Model Building:** Trained and compared several classification models, including Logistic Regression, Random Forest, and XGBoost.
5.  **Hyperparameter Tuning:** Optimized the XGBoost model using `GridSearchCV` to find the best parameters and improve performance.
6.  **Handling Class Imbalance:** Applied the **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data to address the imbalance between churned and non-churned customers, significantly improving the model's recall.
7.  **Model Evaluation:** Assessed the final model using key metrics like Accuracy, Precision, Recall, and F1-Score, focusing on recall for the "Churn" class.
8.  **Deployment:** Saved the final trained model, scaler, and column list using `joblib` and deployed it as an interactive web app using Streamlit Community Cloud.

---

## 📈 Model Performance

The final **Tuned XGBoost model (with SMOTE)** achieved the following performance on the test set, demonstrating a strong ability to identify customers at risk of churning:

| Metric | Score (for Churn = Yes) |
| :--- | :--- |
| **Accuracy** | ~79% |
| **Precision** | ~59% |
| **Recall** | **~69%** |
| **F1-Score** | ~63% |

*The high **Recall** score was the primary goal, as it's more crucial for the business to identify potential churners (even with a few false positives) than to miss them.*

---

## 🚀 How to Run This Project Locally

Follow these steps to set up and run the application on your own machine.

1.  **Clone the Repository:**
    ```bash
    git clone [[YOUR_GITHUB_REPO_LINK](https://github.com/Divyansh-debug/Celebal-Customer-Churn-Prediction-Project.git)]
    cd Customer-Churn-Prediction-App
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in a new tab in your web browser!

---

## 📂 Repository Structure
├── app.py                  # Main Streamlit application script
├── churn_model.joblib      # Trained XGBoost model
├── scaler.joblib           # Scikit-learn scaler object
├── train_columns.joblib    # List of training columns
├── requirements.txt        # Required Python libraries
├── Churn_Celebal_Trial_new.ipynb # Jupyter Notebook with full analysis
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # The dataset
└── README.md               # This file


---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

<div align="center">
  <i>Made with ❤️ and a lot of Python!</i>
</div>
