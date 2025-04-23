import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle



def model_training(filepath='classifier.csv'):
    df = pd.read_csv(filepath)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # splitting data
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # target column ellimination
    X_train, y_train = train_data.drop('Label', axis=1), train_data['Label']
    X_test, y_test = test_data.drop('Label', axis=1), test_data['Label']
    # scaling the training data
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=['pH', 'Temperature', 'GasFlow', 'CH4', 'FeedRate'])

    # scaling the test data
    X_test_scaled = scaler.transform(X_test.to_numpy())
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=['pH', 'Temperature', 'GasFlow', 'CH4', 'FeedRate'])

    y_train_fixed = y_train - 1
    y_test_fixed = y_test - 1

    # Example param grid
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    xgb = XGBClassifier()

    classifier = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,  # ONLY 20 random combinations tested!
        cv=3,  # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all cores
    )

    classifier.fit(X_train_scaled, y_train_fixed)
    return classifier





