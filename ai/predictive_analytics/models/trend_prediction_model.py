import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class TrendPredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def preprocess_data(self):
        # Scale features
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(self.data[['feature1', 'feature2', 'feature3']])

        # Split data into features and target
        X = self.data.drop(['date', 'target'], axis=1)
        y = self.data['target']

        return X, y

    def train(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
        model = LinearRegression()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))

        self.model = model

    def predict(self, trend_data):
        if self.model is None:
            raise Exception("Model not trained yet")

        # Preprocess trend data
        trend_data[['feature1', 'feature2', 'feature3']] = StandardScaler().fit_transform(trend_data[['feature1', 'feature2', 'feature3']])

        # Make prediction
        prediction = self.model.predict(trend_data)

        return prediction
