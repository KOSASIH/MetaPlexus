import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class InsightsModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def preprocess_data(self):
        # Scale features
        scaler = StandardScaler()
        self.data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(self.data[['feature1', 'feature2', 'feature3']])

        return self.data

    def train(self):
        data = self.preprocess_data()

        # Define model
        model = KMeans(n_clusters=5)

        # Train model
        model.fit(data)

        # Evaluate model
        print("Cluster Centers:")
        print(model.cluster_centers_)

        self.model = model

    def get_insights(self):
        if self.model is None:
            raise Exception("Model not trained yet")

        # Get cluster labels
        labels = self.model.labels_

        # Get insights
        insights = pd.DataFrame({
            'cluster': labels,
            'feature1': self.data['feature1'],
            'feature2': self.data['feature2'],
            'feature3': self.data['feature3']
        })

        return insights
