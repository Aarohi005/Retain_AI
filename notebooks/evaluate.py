import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_model(y_test, y_pred, y_prob):
    """
    Evaluate classification model performance with multiple metrics.
    
    Parameters:
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like
        Predicted probabilities for the positive class
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Example usage (only runs if this file is executed directly)
if __name__ == "__main__":
    from data_cleaning import load_and_clean
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    # Load cleaned dataset
    df = load_and_clean("D:/projects/churn analysis/data/European_Bank.csv")

    # Prepare dataset
    df_model = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    evaluate_model(y_test, y_pred, y_prob)
