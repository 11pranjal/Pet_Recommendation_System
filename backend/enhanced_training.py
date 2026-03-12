"""
Enhanced Pet Recommendation System Training
- Trains KNN model for quiz-based recommendations  
- Creates SBERT embeddings for text-based search
- Uses ACTUAL dataset with proper mappings
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path so we can import pet_image_mapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pet_image_mapper import get_pet_image_url

# Paths
CSV_PATH = "dataset/Pet_Recommendation_System.csv"  # KNN training data (preprocessed/normalized)
CSV_PATH_ORIGINAL = "dataset/fully_updated_pet_dataset.csv"  # Original data with real values for display
CSV_PATH_SBERT = "dataset/sbert_refined_data_with_breed_characteristics_gender_full_enhanced.csv"  # SBERT with enhanced characteristics
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# EXACT MAPPINGS FROM YOUR DATASET
PET_TYPE_MAPPING = {
    0: "Bird",
    1: "Cat", 
    2: "Dog",
    3: "Rabbit"
}

BREED_MAPPING = {
    0: "Domestic Shorthair",
    1: "German Shepherd", 
    2: "Labrador",
    3: "Parakeet",
    4: "Persian",
    5: "Pug",
    6: "Rabbit",
    7: "Retriever",
    8: "Spitz"
}

COLOR_MAPPING = {
    0: "Beige",
    1: "Black",
    2: "Black and Red",
    3: "Black and Tan",
    4: "Brown",
    5: "Fawn",
    6: "Gray",
    7: "Green",
    8: "Orange",
    9: "Sable",
    10: "White"
}

SIZE_MAPPING = {
    0: "Large",
    1: "Medium",
    2: "Small"
}

GENDER_MAPPING = {
    0: "Female",
    1: "Male"
}

ENERGY_MAPPING = {
    0: "High",
    1: "Low",
    2: "Moderate"
}

def generate_pet_label(breed, pet_id):
    """Generate a pet label using breed + dataset PetID, e.g. 'German Shepherd #42'"""
    return f"{breed} #{pet_id}"

def generate_pet_image(breed_name, color, size, age_months, pet_id):
    """Generate pet image URL from local petimage folder using the image mapper."""
    url = get_pet_image_url(breed_name, color, size, age_months, pet_id)
    return url  # may be "" if no image found for this breed

def denormalize_age(normalized_age):
    """Convert normalized age back to months"""
    # Exact StandardScaler params computed from fully_updated_pet_dataset.csv
    # mean=81.5571788413, std=57.8377029211
    age = int(normalized_age * 57.8377 + 81.5572)
    return max(1, age)

def denormalize_weight(normalized_weight):
    """Convert normalized weight back to kg"""
    # Exact StandardScaler params computed from fully_updated_pet_dataset.csv
    # mean=6.8641311952, std=8.7905346871
    weight = normalized_weight * 8.7905 + 6.8641
    return round(max(0.01, weight), 2)

def denormalize_days(normalized_days):
    """Convert normalized shelter days to actual days"""
    # Exact StandardScaler params computed from fully_updated_pet_dataset.csv
    # mean=43.5355163728, std=25.5978957150
    days = int(normalized_days * 25.5979 + 43.5355)
    return max(0, days)

def load_and_prepare_dataset():
    """
    Load preprocessed dataset for KNN training + original dataset for display.
    - Pet_Recommendation_System.csv → normalized features for KNN/SBERT
    - fully_updated_pet_dataset.csv → real values (age, weight, etc.) shown to user
    """
    print("Loading preprocessed dataset from:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    
    print(f"Dataset loaded: {len(df)} pets")
    print(f"Columns: {list(df.columns)}")
    
    # Load ORIGINAL dataset for real display values
    print("Loading original dataset from:", CSV_PATH_ORIGINAL)
    original_df = pd.read_csv(CSV_PATH_ORIGINAL)
    
    # Build lookup: PetID → original row (real age, weight, days, size, etc.)
    original_lookup = {}
    for _, row in original_df.iterrows():
        original_lookup[int(row['PetID'])] = row
    print(f"✓ Loaded {len(original_lookup)} pets from original dataset")
    
    # Load enhanced characteristics dataset
    try:
        sbert_df = pd.read_csv(CSV_PATH_SBERT)
        print(f"✓ Loaded enhanced characteristics from SBERT dataset")
        
        # Create mapping of PetID to enhanced characteristics AND pet_details
        enhanced_chars = {}
        pet_details_map = {}
        for _, row in sbert_df.iterrows():
            pet_id = int(row['PetID'])
            enhanced_chars[pet_id] = row['pet_characteristics']
            if 'pet_details' in row and pd.notna(row['pet_details']):
                pet_details_map[pet_id] = row['pet_details']
    except FileNotFoundError:
        print("⚠️  Enhanced dataset not found, using default")
        enhanced_chars = {}
        pet_details_map = {}
    
    # Create pets database
    pets_database = []
    pet_counter = {0: 0, 1: 0, 2: 0, 3: 0}  # Counter for each pet type
    
    for idx, row in df.iterrows():
        pet_type = int(row['PetType'])
        breed_code = int(row['Breed'])
        pet_id = int(row['PetID'])
        
        # Get the ORIGINAL row for this pet (real values for display)
        orig = original_lookup.get(pet_id)
        
        # Generate label using actual PetID from dataset: "German Shepherd #42"
        breed_name = BREED_MAPPING[breed_code]
        pet_name = generate_pet_label(breed_name, pet_id)
        pet_counter[pet_type] += 1
        
        # Use REAL values from original dataset for display
        # Fall back to denormalized values only if original not found
        if orig is not None:
            real_age = int(orig['AgeMonths'])
            real_weight = round(float(orig['WeightKg']), 2)
            real_days = int(orig['TimeInShelterDays'])
            real_size = str(orig['Size'])
            real_gender = 'Female' if str(orig['Gender']).strip().upper() in ('F', 'FEMALE') else 'Male'
            real_color = str(orig['Color']).strip().title()
            real_energy = str(orig.get('Energy level', orig.get('EnergyLevel', ''))).strip().capitalize()
            real_food = str(orig['FoodPreference'])
        else:
            # Fallback: denormalize from preprocessed data
            real_age = denormalize_age(row['AgeMonths'])
            real_weight = denormalize_weight(row['WeightKg'])
            real_days = denormalize_days(row['TimeInShelterDays'])
            real_size = SIZE_MAPPING[int(row['Size'])]
            real_gender = GENDER_MAPPING[int(row['Gender'])]
            real_color = COLOR_MAPPING[int(row['Color'])]
            real_energy = ENERGY_MAPPING[int(row['EnergyLevel'])]
            real_food = row['FoodPreference']
        
        pet_info = {
            'id': pet_id,
            'index': idx,  # Row index for KNN
            'name': pet_name,
            'type': PET_TYPE_MAPPING[pet_type],
            'breed': BREED_MAPPING[breed_code],
            'age_months': real_age,
            'color': real_color,
            'size': real_size,
            'weight_kg': real_weight,
            'vaccinated': bool(row['Vaccinated']),
            'health_condition': "Excellent" if row['HealthCondition'] == 0 else "Good",
            'days_in_shelter': real_days,
            'has_previous_owner': bool(row['PreviousOwner']),
            'gender': real_gender,
            'description': row['pet_details'],
            'shedding_level': int(row['shedding']),
            'food_preference': real_food,
            'meat_consumption': bool(row['MeatConsumption']),
            'kid_friendly': bool(row['kid_friendliness']),
            'energy_level': real_energy,
            'image_url': generate_pet_image(breed_name, real_color, real_size, real_age, pet_id),
            # Add enhanced physical characteristics if available
            'pet_characteristics': enhanced_chars.get(pet_id, ""),
            # Add pet_details from SBERT dataset (personality, behavior traits)
            'pet_details': pet_details_map.get(pet_id, row['pet_details']),
            # Keep raw features for KNN matching (normalized values)
            'raw_features': {
                'PetType': pet_type,
                'Breed': breed_code,
                'AgeMonths': row['AgeMonths'],
                'Color': int(row['Color']),
                'Size': int(row['Size']),
                'WeightKg': row['WeightKg'],
                'Vaccinated': int(row['Vaccinated']),
                'HealthCondition': int(row['HealthCondition']),
                'TimeInShelterDays': row['TimeInShelterDays'],
                'PreviousOwner': int(row['PreviousOwner']),
                'Gender': int(row['Gender']),
                'shedding': int(row['shedding']),
                'MeatConsumption': int(row['MeatConsumption']),
                'kid_friendliness': int(row['kid_friendliness']),
                'EnergyLevel': int(row['EnergyLevel'])
            }
        }
        pets_database.append(pet_info)
    
    print(f"\n✓ Created database with {len(pets_database)} pets:")
    for pet_type, count in pet_counter.items():
        print(f"  - {PET_TYPE_MAPPING[pet_type]}: {count}")
    
    return df, pets_database

def train_knn_model(df):
    """Train KNN model for quiz-based recommendations"""
    print("\n" + "="*50)
    print("Training KNN Model...")
    print("="*50)
    
    # Features to use for matching (user preferences)
    # 9 features: all map to quiz questions (directly or indirectly)
    feature_columns = ['Size', 'EnergyLevel', 'kid_friendliness', 'Vaccinated',
                      'shedding', 'MeatConsumption', 'AgeMonths', 'WeightKg',
                      'HealthCondition']
    
    X = df[feature_columns].values.astype(float)
    y = df.index.values  # Use dataframe index as target
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN
    # Optimal k=5 determined by Elbow Method analysis (best RMSE, odd number)
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric='euclidean')
    knn.fit(X_scaled, y)
    
    # Save model and components
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "knn_scaler.joblib"))
    np.save(os.path.join(MODEL_DIR, "knn_X_scaled.npy"), X_scaled)
    
    # Save feature column names for reference
    with open(os.path.join(MODEL_DIR, "knn_features.txt"), "w") as f:
        f.write(",".join(feature_columns))
    
    print(f"✓ KNN model trained with {len(X)} samples")
    print(f"✓ Feature columns: {feature_columns}")
    
    return knn, scaler, X_scaled

def train_sbert_model(pets_database):
    """Create SBERT embeddings for text-based search with enhanced characteristics"""
    print("\n" + "="*50)
    print("Creating SBERT Embeddings...")
    print("="*50)
    
    # Load enhanced SBERT dataset for better physical characteristics
    try:
        sbert_df = pd.read_csv(CSV_PATH_SBERT)
        print(f"✓ Loaded enhanced SBERT dataset: {len(sbert_df)} pets")
        
        # Create mapping of PetID to enhanced characteristics
        enhanced_chars = {}
        for _, row in sbert_df.iterrows():
            enhanced_chars[int(row['PetID'])] = row['pet_characteristics']
        
    except FileNotFoundError:
        print("⚠️  Enhanced dataset not found, using default characteristics")
        enhanced_chars = {}
    
    # Load SBERT model (lightweight and fast)
    print("Loading SBERT model: all-MiniLM-L6-v2")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create comprehensive text descriptions using real values from original dataset
    pet_texts = []
    for pet in pets_database:
        pet_id = pet['id']
        
        # Build rich description for semantic search
        text = f"{pet['type']} {pet['breed']}. "
        text += f"{pet['gender']} {pet['size'].lower()} size, {pet['color'].lower()} color. "
        text += f"{pet['age_months']} months old, weighs {pet['weight_kg']}kg. "
        text += f"Energy level: {pet['energy_level']}. "
        
        # Diet info (important for search queries like "non-meat pet" or "vegetarian pet")
        if pet.get('meat_consumption'):
            text += "Meat-based diet. "
        else:
            text += "Non-meat diet, herbivore. "
        
        # Food details from original dataset
        if pet.get('food_preference'):
            text += f"Diet: {pet['food_preference']}. "
        
        # Kid friendliness
        if pet['kid_friendly']:
            text += "Great with kids, family-friendly. "
        else:
            text += "Not ideal for young kids. "
        
        # Vaccination status
        if pet['vaccinated']:
            text += "Fully vaccinated. "
        else:
            text += "Not yet vaccinated. "
        
        # Health condition
        health = pet.get('health_condition', 'Unknown')
        if health == "Excellent":
            text += "Excellent health condition. "
        elif health == "Good":
            text += "Good health condition. "
        else:
            text += f"Health condition: {health}. "
        
        # Previous owner
        if pet.get('has_previous_owner'):
            text += "Has had a previous owner, previously adopted. "
        else:
            text += "No previous owner, first-time adoption. "
        
        # Time in shelter
        days = pet.get('days_in_shelter', 0)
        if days <= 10:
            text += f"Recently arrived at shelter, {days} days. "
        elif days <= 30:
            text += f"Been in shelter for {days} days. "
        else:
            text += f"Long-term shelter resident, {days} days in shelter. "
        
        # Shedding info (all levels)
        shed = pet.get('shedding_level', 0)
        shedding_desc = {0: "No shedding", 1: "Very low shedding", 2: "Low shedding", 
                         3: "Moderate shedding", 4: "High shedding", 5: "Heavy shedding"}
        text += f"{shedding_desc.get(shed, 'Unknown shedding')}. "
        
        # Add personality description
        text += f"Personality: {pet['description']}. "
        
        # Add enhanced physical characteristics if available (fixed: use 'id' not 'pet_id')
        if pet_id in enhanced_chars:
            text += f"Physical characteristics: {enhanced_chars[pet_id]}. "
        elif pet.get('pet_characteristics'):
            text += f"Physical characteristics: {pet['pet_characteristics']}. "
        
        pet_texts.append(text)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(pet_texts)} pets...")
    embeddings = sbert_model.encode(pet_texts, show_progress_bar=True, batch_size=32)
    
    # Save embeddings
    np.save(os.path.join(MODEL_DIR, "sbert_embeddings.npy"), embeddings)
    
    # Save model
    sbert_model.save(os.path.join(MODEL_DIR, "sbert_model"))
    
    print(f"✓ SBERT embeddings created: shape {embeddings.shape}")
    
    return embeddings

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🐾 PET RECOMMENDATION SYSTEM - ENHANCED TRAINING 🐾")
    print("="*60)
    
    # Load dataset
    df, pets_database = load_and_prepare_dataset()
    
    # Save pets database
    db_path = os.path.join(MODEL_DIR, "pets_database.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(pets_database, f)
    print(f"\n✓ Saved pets database to: {db_path}")
    
    # Train KNN model (quiz-based)
    knn, scaler, X_scaled = train_knn_model(df)
    
    # Train SBERT model (text-based)
    sbert_embeddings = train_sbert_model(pets_database)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFiles created in '{MODEL_DIR}/':")
    print("  📁 pets_database.pkl      - Pet information database")
    print("  🤖 knn_model.joblib       - Quiz recommendation model")
    print("  📏 knn_scaler.joblib      - Feature scaler")
    print("  📊 knn_X_scaled.npy       - Scaled feature matrix")
    print("  📝 knn_features.txt       - Feature column names")
    print("  🔤 sbert_embeddings.npy   - Text embeddings")
    print("  🧠 sbert_model/           - SBERT model directory")
    
    # Display sample pets
    print("\n" + "="*60)
    print("📋 SAMPLE PETS FROM DATABASE:")
    print("="*60)
    
    # Show 2 pets from each type
    types_shown = {0: 0, 1: 0, 2: 0, 3: 0}
    for pet in pets_database:
        pet_type_code = pet['raw_features']['PetType']
        if types_shown[pet_type_code] < 2:
            print(f"\n🐾 {pet['name']} — {pet['type']}")
            print(f"   {pet['gender']}, {pet['age_months']} months, {pet['size']}, {pet['color']}")
            print(f"   Energy: {pet['energy_level']} | Kid-friendly: {pet['kid_friendly']}")
            print(f"   Vaccinated: {pet['vaccinated']} | Health: {pet['health_condition']}")
            print(f"   {pet['description'][:100]}...")
            types_shown[pet_type_code] += 1
        
        if sum(types_shown.values()) >= 8:
            break
    
    print("\n" + "="*60)
    print("🎉 Ready to start recommending pets! 🎉")
    print("="*60)

if __name__ == "__main__":
    main()
