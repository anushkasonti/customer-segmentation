**CUSTOMER SPENDING CATEGORIZATION USING NAIVE BAYES**
(Machine Learning)

**Project Overview:**
- This project analyzes a retail sales dataset to categorize customers into Low, Medium, and High spenders using the Naive Bayes classification algorithm. 
- The workflow includes data preprocessing, feature engineering, classification model training, and evaluation with a confusion matrix and performance metrics.
- This application helps retailers gain insights into customer purchasing behavior.

**Features:**
- Load and clean retail transaction data
- Categorize customers into spending tiers using quantiles
- Feature engineering with one-hot encoding
- Train a Naive Bayes classifier on engineered features
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix
- Visualize results with heatmaps using Seaborn
- Optional: Use cross-validation for better generalization (commented version included)

**Output:**
- Accuracy of 0.97 indicates that 97% of predictions matched the actual spending categories.
- The Confusion Matrix indicates:
  - All 41 low spenders were correctly classified.
  - All 41 medium spenders were correctly classified.
  - Out of 38 high spenders, 34 were correctly classified, while 4 were misclassified as medium.
- This shows the model works accurately, with only minor misclassification between very similar groups of spending.

**Tech Stack:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn
