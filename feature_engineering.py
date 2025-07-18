import pandas as pd


def feature_engineering(df):
    print("Columns in the DataFrame:", df.columns.tolist())
    df = pd.get_dummies(df, columns=['Gender', 'Product Category'], drop_first=True)
    return df


if __name__ == "__main__":
    from categorize_customers import categorize_by_spending
    from load_data import load_and_preprocess

    file_path = 'retail_sales_dataset.csv'
    customer_data = load_and_preprocess(file_path)
    categorized_data = categorize_by_spending(customer_data)

    feature_data = feature_engineering(categorized_data)
    print(feature_data.head())
