# 🛍️ Online Shoppers Intention — Interactive Dashboard

This project predicts **whether an online shopper will make a purchase** based on their browsing behavior.  
We leverage **Logistic Regression**, **Random Forest**, and a **Hyperparameter-Tuned Random Forest** model to achieve high accuracy and balanced performance.

🔗 **Live Demo:** [Streamlit App](https://online-shoppers-prediction.streamlit.app/)

📊 **Dataset:** [Kaggle – Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset/data)

---

## 🧾 Dataset Overview

The dataset contains session-level attributes including:

- 🕒 Page visit durations  
- 🚪 Bounce rates & exit rates  
- 🖥️ OS, browser, and region  
- 🌐 Traffic type & weekend behavior  
- 💰 **Target Variable:** `Revenue` (1 = Buy, 0 = Not Buy)

---

## 📈 App Preview

| Home Page | Prediction Interface |
|-----------|----------------------|
| ![Home Page](assets/home.png) | ![Prediction](assets/predict.png) |

---

##  Data Preprocessing & Exploration

- Checked for missing values, duplicates, and data types.  
- Visualized data with histograms, box plots, and a correlation matrix.  
- Encoded categorical features & scaled numerical values.  
- Split dataset into training and test sets (80/20).

```python
import pandas as pd

df = pd.read_csv("online_shoppers_intention.csv")
df.info()
df.describe()
df.isnull().sum()
```

---

##  Baseline Model — Random Forest

- Trained a default Random Forest Classifier.  
- Evaluated model performance using accuracy, precision, recall, and F1-score.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

##  Hyperparameter Tuning — GridSearchCV

We optimized the Random Forest model to improve performance and reduce overfitting.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```



##  Interactive Dashboard (Streamlit)

The dashboard allows users to:

- 📥 Upload a dataset  
- ⚙️ Train & test different models  
- 📊 Visualize performance metrics  
- 🧮 Make predictions for new data in real time  

```bash
# Clone repo
git clone https://github.com/your-username/online-shoppers-prediction.git
cd online-shoppers-prediction

# Optional: create virtual env
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard/app.py
```

---

## 📂 Project Structure

```
.
├── dashboard/
│   ├── app.py
│   ├── components/
│   ├── utils/
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── data/
│   └── online_shoppers_intention.csv
├── assets/
│   ├── home.png
│   └── predict.png
├── requirements.txt
└── README.md
```

