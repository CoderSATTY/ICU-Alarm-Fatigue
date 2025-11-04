import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

class TransformerClassifier:
    def __init__(self, num_classes, num_features, head_dim=32, num_heads=4, num_layers=2, ff_dim=128, dropout=0.1):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.num_classes = num_classes
        self.num_features = num_features
        self.history = None
        self.model = self._build_model(head_dim, num_heads, num_layers, ff_dim, dropout)

    def _build_model(self, head_dim, num_heads, num_layers, ff_dim, dropout):
        inputs = keras.Input(shape=(self.num_features,))
        x = layers.Dense(head_dim * num_heads)(inputs)
        x = layers.Reshape((1, head_dim * num_heads))(x)
        
        for _ in range(num_layers):
            attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            ff = layers.Dense(ff_dim, activation='relu')(x)
            ff = layers.Dense(head_dim * num_heads)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(dropout)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                      validation_split=validation_split, verbose=0)

    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

if __name__ == '__main__':
    df = pd.read_csv('C:\\IITI\\AIML\\ICU_Alarm_Fatigue\\VAPNOVAPDataset.xlsx - Sheet1.csv')
    feature_cols = ['PEAK', 'PMEAN', 'PEEP1', 'I', 'E', 'FTOT', 'VTE', 'VETOT', 'PR', 'SpO2']
    
    X = df[feature_cols].values
    y = df['Label'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data size: {len(X_train)}")
    print(f"Testing data size: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_encoded))}\n")
    
    num_classes = len(np.unique(y_encoded))
    classifier = TransformerClassifier(num_classes=num_classes, num_features=len(feature_cols))
    
    print("Training Transformer Model on 80% data...")
    classifier.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    y_pred = classifier.predict(X_test_scaled)
    
    print("\n" + "="*70)
    print("TRAINING REPORT - TRANSFORMER CLASSIFIER")
    print("="*70)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nOverall Metrics (on 20% test data):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\n" + "="*70)
    print("Training History Summary:")
    print("="*70)
    print(f"Final Training Loss: {classifier.history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {classifier.history.history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {classifier.history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {classifier.history.history['val_accuracy'][-1]:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(classifier.history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(classifier.history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(classifier.history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(classifier.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()