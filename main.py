# main.py
from load_data import load_and_preprocess
from categorize_customers import categorize_by_spending
from feature_engineering import feature_engineering
from train_model import train_naive_bayes


def main():
    file_path = 'retail_sales_dataset.csv'
    customer_data = load_and_preprocess(file_path)

    categorized_data = categorize_by_spending(customer_data)

    feature_data = feature_engineering(categorized_data)

    X = feature_data.drop('Spending_Category', axis=1)
    y = feature_data['Spending_Category']

    train_naive_bayes(X, y)


if __name__ == "__main__":
    main()
