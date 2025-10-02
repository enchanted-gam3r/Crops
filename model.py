import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Loading data
try:
    df = pd.read_csv('../crop_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

#Data Prep
TARGET = 'Crop'
FEATURES = [col for col in df.columns if col not in [TARGET, 'Yield_q_per_ha', 'MarketPrice_Rs_per_q']]

X = df[FEATURES]
y = df[TARGET]

# Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

categorical_features = ['Irrigation', 'Season']
numerical_features = [col for col in FEATURES if col not in categorical_features]

#Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

#Piplining the model and model training (using Random Forest Algorithm)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print("\nTraining the Random Forest model...")
model_pipeline.fit(X_train, y_train)
print("Model training completed.")

print("\nEvaluating model performance...")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

#To save the model for further use
joblib.dump(model_pipeline, 'crop_suggestion_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\nModel and label encoder have been saved successfully as 'crop_suggestion_model.joblib' and 'label_encoder.joblib'.")
