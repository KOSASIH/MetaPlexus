import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class UserBehaviorModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def preprocess_data(self):
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.data[['session_duration', 'pages_visited', 'time_spent']] = imputer.fit_transform(self.data[['session_duration', 'pages_visited', 'time_spent']])

        # Scale features
        scaler = StandardScaler()
        self.data[['session_duration', 'pages_visited', 'time_spent']] = scaler.fit_transform(self.data[['session_duration', 'pages_visited', 'time_spent']])

        # Encode categorical variables
        self.data['device_type'] = self.data['device_type'].astype('category')
        self.data['device_type'] = self.data['device_type'].cat.codes

        self.data['browser_type'] = self.data['browser_type'].astype('category')
        self.data['browser_type'] = self.data['browser_type'].cat.codes

        # Split data into features and target
        X = self.data.drop(['user_id', 'target'], axis=1)
        y = self.data['target']

        return X, y

    def train(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define pipeline
        pipeline = Pipeline([
            ('rfc', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = pipeline.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        self.model = pipeline

    def predict(self, user_data):
        if self.model is None:
            raise Exception("Model not trained yet")

        # Preprocess user data
        user_data[['session_duration', 'pages_visited', 'time_spent']] = SimpleImputer(strategy='mean').fit_transform(user_data[['session_duration', 'pages_visited', 'time_spent']])
        user_data[['session_duration', 'pages_visited', 'time_spent']] = StandardScaler().fit_transform(user_data[['session_duration', 'pages_visited', 'time_spent']])
        user_data['device_type'] = user_data['device_type'].astype('category')
        user_data['device_type'] = user_data['device_type'].cat.codes
        user_data['browser_type'] = user_data['browser_type'].astype('category')
        user_data['browser_type'] = user_data['browser_type'].cat.codes

        # Make prediction
        prediction = self.model.predict(user_data)

        return prediction
