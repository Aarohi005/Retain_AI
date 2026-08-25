import pandas as pd

def load_and_clean(path="D:/projects/churn analysis/data/European_Bank.csv"):
    df = pd.read_csv(path)
    df.drop(['CustomerId','Surname','Year'], axis=1, inplace=True)
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    return df

# Keep Geography for EDA, encode later for ML
df = load_and_clean()

if __name__ == "__main__":
    print(df.head())
