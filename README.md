# Online-Shoppers-Prediction
This project aims to predict whether an online shopper will buy or not based on their browsing behavior. We use machine learning models, including Logistic Regression, Random Forest, and a Hyperparameter-Tuned Random Forest to optimize prediction accuracy. 


Dataset  
The dataset is from [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset/data) . It has various user session attributes, such as:
- Page visit durations
- Bounce rates & exit rates
- Operating System, Browser, and Region
- Traffic type & weekend behavior
- Revenue (Target Variable: 1 = Buy, 0 = Not Buy)





1️⃣ Data Preprocessing & Exploration  
- Load dataset and check for missing values, duplicates, and data types.  
- Visualize distributions (histograms, box plots, correlation matrix).  
- Clean & preprocess data:  

2️⃣ Baseline Model - Random Forest  
- Train a default Random Forest classifier.  
- Evaluate model performance using accuracy, precision, recall, and F1-score.  

3️⃣ Hyperparameter Tuning - GridSearchCV  
- Optimize Random Forest parameters for better performance:
  - n_estimators: [100, 200, 300]  
  - max_depth: [None, 10, 20]  
  - min_samples_split: [2, 5, 10]  
  - min_samples_leaf: [1, 2, 4]  
- GridSearchCV performs an exhaustive search over all parameter combinations.  
- The best combination is selected based on cross-validation performance.  

4️⃣ Model Comparison & Final Prediction  
- Compare Baseline vs. Tuned Random Forest performance.  
- Make a final prediction: Will the user buy or not?  

- The hyperparameter-tuned model performed the best, balancing bias and variance while improving overall accuracy.  
- This model is now optimized for predicting user purchase behavior.


## Interactive Dashboard

An interactive dashboard is included to train, test, and visualize model performance on the Online Shoppers dataset.

Quick start:

1. Create and activate a virtual environment (optional)
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.\\.venv\\Scripts\\Activate.ps1`
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Install dependencies
   - `pip install -r requirements.txt`

3. (Optional) Prepare cleaned CSV like the notebook
   - `python tools/prepare_cleaned_csv.py --input online_shoppers_intention.csv --output online_shoppers_intention_cleaned.csv`
   - The dashboard will auto-use `online_shoppers_intention_cleaned.csv` if present; otherwise it applies the same preprocessing in-app.

4. Launch the dashboard
   - `streamlit run dashboard/app.py`

What you can do:
- Use a clean, user-friendly Predict tab with simple fields and tooltips.
- Choose a model (Logistic Regression or Random Forest) and train with a configurable split.
- View metrics (Accuracy, Precision, Recall, F1, ROC AUC), confusion matrix, ROC and PR curves.
- Inspect feature importance (tree-based or coefficients for logistic regression).
- See notebook-style visuals (class balance, correlation, distributions) with explanations.

