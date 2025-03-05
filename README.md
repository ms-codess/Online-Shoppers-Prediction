# Online-Shoppers-Prediction
This project aims to predict whether an online shopper will buy or not based on their browsing behavior. We use machine learning models, including Logistic Regression, Random Forest, and a Hyperparameter-Tuned Random Forest to optimize prediction accuracy. \


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

