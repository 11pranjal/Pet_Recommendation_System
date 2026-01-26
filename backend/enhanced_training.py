"""
Enhanced Pet Recommendation System Training
- Trains KNN model for quiz-based recommendations  
- Creates SBERT embeddings for text-based search
- Uses ACTUAL dataset with proper mappings
"""

import os
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

# Paths
CSV_PATH = "dataset/final_refined_pet.csv"  # KNN training data with all fields
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

# Pet name lists for each type
DOG_NAMES = ["Max", "Bella", "Charlie", "Luna", "Cooper", "Daisy", "Rocky", "Molly", 
             "Buddy", "Sadie", "Duke", "Maggie", "Bear", "Lucy", "Tucker", "Bailey",
             "Jack", "Sophie", "Toby", "Chloe", "Zeus", "Lily", "Oliver", "Penny",
             "Bentley", "Zoe", "Winston", "Lola", "Diesel", "Stella"]

CAT_NAMES = ["Whiskers", "Mittens", "Shadow", "Luna", "Oliver", "Bella", "Simba", "Chloe",
             "Leo", "Nala", "Milo", "Cleo", "Felix", "Lucy", "Tiger", "Princess",
             "Smokey", "Angel", "Jasper", "Kitty", "Oscar", "Misty", "Salem", "Patches",
             "Ginger", "Boots", "Buddy", "Socks", "Pumpkin", "Fluffy"]

BIRD_NAMES = ["Tweety", "Rio", "Kiwi", "Sunny", "Blu", "Polly", "Chirpy", "Sky",
              "Mango", "Coco", "Peach", "Charlie", "Buddy", "Angel", "Rainbow",
              "Skittles", "Peanut", "Cookie", "Daisy", "Lucky", "Happy", "Pepper",
              "Lemon", "Berry", "Sweetie", "Echo", "Ruby", "Sapphire"]

RABBIT_NAMES = ["Thumper", "Clover", "Cottontail", "Hoppy", "Fluffy", "Snowball", 
                "Patches", "Oreo", "Bunny", "Cotton", "Marshmallow", "Nibbles",
                "Cinnamon", "Hazel", "Willow", "Daisy", "Flopsy", "Peter",
                "Honey", "Maple", "Smokey", "Pepper", "Sugar", "Butterscotch"]

def generate_pet_name(pet_type, pet_id):
    """Generate consistent pet name based on type and ID"""
    if pet_type == 2:  # Dog
        return DOG_NAMES[pet_id % len(DOG_NAMES)] + str((pet_id // len(DOG_NAMES)) if pet_id >= len(DOG_NAMES) else "")
    elif pet_type == 1:  # Cat
        return CAT_NAMES[pet_id % len(CAT_NAMES)] + str((pet_id // len(CAT_NAMES)) if pet_id >= len(CAT_NAMES) else "")
    elif pet_type == 0:  # Bird
        return BIRD_NAMES[pet_id % len(BIRD_NAMES)] + str((pet_id // len(BIRD_NAMES)) if pet_id >= len(BIRD_NAMES) else "")
    elif pet_type == 3:  # Rabbit
        return RABBIT_NAMES[pet_id % len(RABBIT_NAMES)] + str((pet_id // len(RABBIT_NAMES)) if pet_id >= len(RABBIT_NAMES) else "")
    return "Pet" + str(pet_id)

def generate_pet_image(pet_type, pet_id, seed):
    """Generate pet image URL using placeholder service"""
    # Using Lorem Picsum with different seeds for variety
    # You can replace with actual image URLs or upload system
    
    if pet_type == 2:  # Dog
        # Use Unsplash Source for dogs (search term: dog)
        return f"https://source.unsplash.com/400x400/?dog,puppy&sig={seed}"
    elif pet_type == 1:  # Cat
        # Use Unsplash Source for cats (search term: cat)
        return f"https://source.unsplash.com/400x400/?cat,kitten&sig={seed}"
    elif pet_type == 0:  # Bird
        # Use Unsplash Source for birds (search term: bird)
        return f"https://source.unsplash.com/400x400/?bird,parrot&sig={seed}"
    elif pet_type == 3:  # Rabbit
        # Use Unsplash Source for rabbits (search term: rabbit)
        return f"https://source.unsplash.com/400x400/?rabbit,bunny&sig={seed}"
    
    # Fallback
    return f"https://source.unsplash.com/400x400/?pet,animal&sig={seed}"

def denormalize_age(normalized_age):
    """Convert normalized age back to approximate months"""
    # Based on typical normalization: (x - mean) / std
    # Approximate: mean~24, std~15 months
    age = int(normalized_age * 15 + 24)
    return max(1, age)

def denormalize_weight(normalized_weight):
    """Convert normalized weight back to approximate kg"""
    # Approximate: mean~15kg, std~10kg
    weight = normalized_weight * 10 + 15
    return round(max(0.5, weight), 1)

def denormalize_days(normalized_days):
    """Convert normalized shelter days to actual days"""
    days = int(normalized_days * 30 + 45)
    return max(0, days)

def load_and_prepare_dataset():
    """Load dataset and create pets database with readable info"""
    print("Loading dataset from:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    
    print(f"Dataset loaded: {len(df)} pets")
    print(f"Columns: {list(df.columns)}")
    
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
        
        # Generate unique name based on pet type
        pet_name = generate_pet_name(pet_type, pet_counter[pet_type])
        pet_counter[pet_type] += 1
        
        # Generate unique seed for consistent images
        image_seed = pet_id * 1000 + pet_type * 100
        
        pet_info = {
            'id': pet_id,
            'index': idx,  # Row index for KNN
            'name': pet_name,
            'type': PET_TYPE_MAPPING[pet_type],
            'breed': BREED_MAPPING[breed_code],
            'age_months': denormalize_age(row['AgeMonths']),
            'color': COLOR_MAPPING[int(row['Color'])],
            'size': SIZE_MAPPING[int(row['Size'])],
            'weight_kg': denormalize_weight(row['WeightKg']),
            'vaccinated': bool(row['Vaccinated']),
            'health_condition': "Excellent" if row['HealthCondition'] == 0 else "Good",
            'days_in_shelter': denormalize_days(row['TimeInShelterDays']),
            'has_previous_owner': bool(row['PreviousOwner']),
            'gender': GENDER_MAPPING[int(row['Gender'])],
            'description': row['pet_details'],
            'shedding_level': int(row['shedding']),
            'food_preference': row['FoodPreference'],
            'meat_consumption': bool(row['MeatConsumption']),
            'kid_friendly': bool(row['kid_friendliness']),
            'energy_level': ENERGY_MAPPING[int(row['EnergyLevel'])],
            'image_url': generate_pet_image(pet_type, pet_id, image_seed),
            # Add enhanced physical characteristics if available
            'pet_characteristics': enhanced_chars.get(pet_id, ""),
            # Add pet_details from SBERT dataset (personality, behavior traits)
            'pet_details': pet_details_map.get(pet_id, row['pet_details']),  # Use SBERT details if available, else use KNN details
            # Keep raw features for KNN matching
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
    feature_columns = ['Size', 'EnergyLevel', 'kid_friendliness', 'Vaccinated',
                      'shedding', 'MeatConsumption', 'AgeMonths', 'WeightKg']
    
    X = df[feature_columns].values.astype(float)
    y = df.index.values  # Use dataframe index as target
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN
    # Use more neighbors for better recommendations
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", metric='euclidean')
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
    
    # Create comprehensive text descriptions
    pet_texts = []
    for pet in pets_database:
        # Build rich description for semantic search
        text = f"{pet['type']} {pet['breed']} named {pet['name']}. "
        text += f"{pet['gender']} {pet['size'].lower()} size, {pet['color'].lower()} color. "
        text += f"{pet['age_months']} months old, weighs {pet['weight_kg']}kg. "
        text += f"Energy level: {pet['energy_level']}. "
        
        if pet['kid_friendly']:
            text += "Great with kids. "
        if pet['vaccinated']:
            text += "Fully vaccinated. "
        if pet['health_condition'] == "Excellent":
            text += "Excellent health. "
        
        # Add personality description
        text += f"Personality: {pet['description']}. "
        
        # Add enhanced physical characteristics if available
        if pet.get('pet_id') in enhanced_chars:
            text += f"Physical characteristics: {enhanced_chars[pet['pet_id']]}."
        
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
            print(f"\n🐾 {pet['name']} - {pet['type']} ({pet['breed']})")
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
