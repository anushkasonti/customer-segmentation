import pandas as pd


def categorize_by_spending(customer_data):
    customer_data['Spending_Category'] = pd.qcut(customer_data['Total_Spent'], q=3,
                                                 labels=['Low-spenders', 'Medium-spenders', 'High-spenders'])
    return customer_data


if __name__ == "__main__":
    from load_data import load_and_preprocess

    file_path = 'retail_sales_dataset.csv'
    customer_data = load_and_preprocess(file_path)

    categorized_data = categorize_by_spending(customer_data)
    print(categorized_data.head())
