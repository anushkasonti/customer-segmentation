# # Original code
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report, accuracy_score
#
#
# def train_naive_bayes(X, y):
#     # Split into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     # Initialize and train the model
#     nb = GaussianNB()
#     nb.fit(X_train, y_train)
#
#     # Predict on the test set
#     y_pred = nb.predict(X_test)
#
#     # Print the evaluation results
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#
#     return nb
#
#
# if __name__ == "__main__":
#     from feature_engineering import feature_engineering
#     from categorize_customers import categorize_by_spending
#     from load_data import load_and_preprocess
#
#     # Load and preprocess the data
#     file_path = 'retail_sales_dataset.csv'
#     customer_data = load_and_preprocess(file_path)
#     categorized_data = categorize_by_spending(customer_data)
#     feature_data = feature_engineering(categorized_data)
#
#     # Separate features and labels
#     X = feature_data.drop('Spending_Category', axis=1)
#     y = feature_data['Spending_Category']
#
#     # Train Naive Bayes model
#     train_naive_bayes(X, y)

# with confusion matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Heatmap')
    plt.show()

def train_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm, nb.classes_)

    return nb



if __name__ == "__main__":
    from feature_engineering import feature_engineering
    from categorize_customers import categorize_by_spending
    from load_data import load_and_preprocess

    file_path = 'retail_sales_dataset.csv'
    customer_data = load_and_preprocess(file_path)
    categorized_data = categorize_by_spending(customer_data)
    feature_data = feature_engineering(categorized_data)

    X = feature_data.drop('Spending_Category', axis=1)
    y = feature_data['Spending_Category']

    train_naive_bayes(X, y)

# With cross validation

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report
#
#
# def train_naive_bayes(X, y):
#     nb = GaussianNB()
#
#     # Cross-validation
#     scores = cross_val_score(nb, X, y, cv=5)  # 5-fold cross-validation
#     print("Cross-validation scores:", scores)
#     print("Mean accuracy:", scores.mean())
#
#     # Fit the model on the entire dataset
#     nb.fit(X, y)
#
#     # Predict on the same dataset for reporting (not recommended in practice)
#     y_pred = nb.predict(X)
#
#     # Print the evaluation results
#     print(classification_report(y, y_pred))
#
#
# if __name__ == "__main__":
#     # Load and preprocess the data
#     from feature_engineering import feature_engineering
#     from categorize_customers import categorize_by_spending
#     from load_data import load_and_preprocess
#
#     file_path = 'retail_sales_dataset.csv'
#     customer_data = load_and_preprocess(file_path)
#     categorized_data = categorize_by_spending(customer_data)
#     feature_data = feature_engineering(categorized_data)
#
#     # Separate features and labels
#     X = feature_data.drop('Spending_Category', axis=1)
#     y = feature_data['Spending_Category']
#
#     # Train Naive Bayes model
#     train_naive_bayes(X, y)
