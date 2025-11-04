import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import io
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
class CosineSimilarityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.embeddings = None
        self.labels = None
        self.feature_columns = None

    def fit(self, df, feature_columns, label_column='Label'):
        self.feature_columns = feature_columns
        features = df[self.feature_columns]
        scaled_features = self.scaler.fit_transform(features)
        
        self.embeddings = scaled_features
        self.labels = df[label_column].values

    def predict(self, data_point):
        if self.embeddings is None or self.labels is None:
            raise RuntimeError("Predictor is not fitted. Call fit() or load() first.")

        if isinstance(data_point, dict):
            data_point = pd.Series(data_point)
        
        data_point_features = data_point[self.feature_columns].values.reshape(1, -1)
        scaled_data_point = self.scaler.transform(data_point_features)
        similarities = cosine_similarity(scaled_data_point, self.embeddings)
        most_similar_index = np.argmax(similarities)
        
        return self.labels[most_similar_index]

    def save(self, filepath):
        data_to_save = {
            'scaler': self.scaler,
            'embeddings': self.embeddings,
            'labels': self.labels,
            'feature_columns': self.feature_columns
        }
        joblib.dump(data_to_save, filepath)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        
        saved_data = joblib.load(filepath)
        predictor = cls()
        predictor.scaler = saved_data['scaler']
        predictor.embeddings = saved_data['embeddings']
        predictor.labels = saved_data['labels']
        predictor.feature_columns = saved_data['feature_columns']
        return predictor

if __name__ == '__main__':

    df = pd.read_csv('C:\\IITI\\AIML\\ICU_Alarm_Fatigue\\VAPNOVAPDataset.xlsx - Sheet1.csv')
    feature_cols = ['PEAK', 'PMEAN', 'PEEP1', 'I', 'E', 'FTOT', 'VTE', 'VETOT', 'PR', 'SpO2']
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

    print(f"Training data size: {len(train_df)}")
    print(f"Testing data size: {len(test_df)}")

    predictor = CosineSimilarityPredictor()
    predictor.fit(train_df, feature_columns=feature_cols, label_column='Label')
    print("\nPredictor has been fitted on the 80% training data.")

    embedding_filepath = 'embedding_space.joblib'
    predictor.save(embedding_filepath)
    print(f"Embedding space (database) saved locally to '{embedding_filepath}'")

    loaded_predictor = CosineSimilarityPredictor.load(embedding_filepath)
    print("Embedding space loaded from local file.")

    y_true = test_df['Label']
    y_pred = [loaded_predictor.predict(row) for index, row in test_df.iterrows()]

    print("\n--- Evaluation on 20% Test Data ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


