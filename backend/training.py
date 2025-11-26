import os
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# Path to CSV 
CSV_PATH = "dataset/final_refined_pet.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Features order MUST match the frontend encoding (same order as used to encode user answers)
FEATURE_COLUMNS = [
    'PetID',
    'Breed',
    'AgeMonths',
    'Color',
    'Size',
    'WeightKg',
    'Vaccinated',
    'HealthCondition',
    'TimeInShelterDays',
    'PreviousOwner',
    'Gender',
    'shedding',
    'MeatConsumption',
    'kid_friendliness',
    'EnergyLevel'    # 0..2
]

LABEL_COLUMN ='PetType'  # e.g., 'Dog', 'Cat', 'Rabbit', ...

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Run your notebook to produce CSV or copy it here.")
    df = pd.read_csv(path)
    # Optionally assert columns exist
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df

def train():
    df = load_dataset(CSV_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")
    joblib.dump(knn, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("Saved model and scaler to", MODEL_DIR)

if __name__ == "__main__":
    train()