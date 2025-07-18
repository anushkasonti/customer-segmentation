import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna()

    customer_data = df.groupby('Customer ID').agg({
        'Total Amount': 'sum',
        'Transaction ID': 'count',
        'Gender': 'first',
        'Product Category': 'first'
    }).reset_index()

    customer_data.rename(columns={'Total Amount': 'Total_Spent', 'Transaction ID': 'Num_Transactions'}, inplace=True)

    customer_data = customer_data.drop(columns=['Customer ID'])

    customer_data = customer_data.sample(frac=0.40, random_state=42)

    return customer_data

if __name__ == "__main__":
    file_path = 'retail_sales_dataset.csv'
    customer_data = load_and_preprocess(file_path)
    print(customer_data.head())
