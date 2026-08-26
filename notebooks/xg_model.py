import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from xgboost import plot_importance
import os
from data_cleaning import load_and_clean   # reuse your cleaning function

def train_xgb_model(df):
    """
    Train an XGBoost model on the churn dataset.
    
    Parameters:
    df : pandas.DataFrame
        Cleaned dataset
    
    Returns:
    model : trained XGBClassifier
    X_test, y_test : test features and labels
    y_pred : predictions on test set
    y_prob : predicted probabilities for ROC-AUC
    """
    # Prepare dataset
    df_model = pd.get_dummies(df.copy(), columns=['Geography'], drop_first=True)
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Build model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='logloss'
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_prob


# Example usage (only runs if file is executed directly)
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "European_Bank.csv")
    df = load_and_clean(DATA_PATH)
    model, X_test, y_test, y_pred, y_prob = train_xgb_model(df)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature importance plot
    plot_importance(model)
    plt.show()
