# model building

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()