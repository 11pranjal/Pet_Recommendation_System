"""
Pet Recommendation Engine
- KNN-based quiz recommendations with confidence scores
- SBERT-based text search
- Returns pet names and breeds (not numbers!)
"""

import numpy as np
import joblib
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PetRecommendationEngine:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.pets_database = None
        self.knn_model = None
        self.knn_scaler = None
        self.sbert_model = None
        self.sbert_embeddings = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load all models and data"""
        print("🔄 Loading recommendation models...")
        
        # Load pets database
        pets_db_path = os.path.join(self.model_dir, "pets_database.pkl")
        if not os.path.exists(pets_db_path):
            raise FileNotFoundError(f"Pets database not found! Please run training first.")
        
        with open(pets_db_path, "rb") as f:
            self.pets_database = pickle.load(f)
        print(f"✓ Loaded {len(self.pets_database)} pets from database")
        
        # Load KNN model
        self.knn_model = joblib.load(os.path.join(self.model_dir, "knn_model.joblib"))
        self.knn_scaler = joblib.load(os.path.join(self.model_dir, "knn_scaler.joblib"))
        print("✓ Loaded KNN model")
        
        # Load feature columns
        with open(os.path.join(self.model_dir, "knn_features.txt"), "r") as f:
            self.feature_columns = f.read().strip().split(",")
        print(f"✓ KNN features: {self.feature_columns}")
        
        # Load SBERT model and embeddings
        self.sbert_model = SentenceTransformer(os.path.join(self.model_dir, "sbert_model"))
        self.sbert_embeddings = np.load(os.path.join(self.model_dir, "sbert_embeddings.npy"))
        print("✓ Loaded SBERT model and embeddings")
        
        print("✅ All models loaded successfully!\n")
    
    def recommend_from_quiz(self, user_answers, top_k=5):
        """
        Recommend pets based on quiz answers
        
        Args:
            user_answers: dict with quiz answers (all 11 questions)
            top_k: number of recommendations
            
        Returns:
            list of pet recommendations with confidence scores
        """
        # Convert user answers to feature vector
        feature_vector = self._quiz_to_features(user_answers)
        
        # Scale features
        feature_scaled = self.knn_scaler.transform([feature_vector])
        
        # Get more candidates for filtering (request 3x to allow filtering)
        n_candidates = min(top_k * 3, len(self.pets_database))
        distances, indices = self.knn_model.kneighbors(feature_scaled, n_neighbors=n_candidates)
        
        # Convert distances to confidence percentage (0-100%)
        if distances[0].max() > 0:
            max_dist = distances[0].max()
            confidences = [(1 - (dist / max_dist)) * 100 for dist in distances[0]]
        else:
            confidences = [100.0] * len(distances[0])
        
        # Get pet recommendations with filtering
        recommendations = []
        for pet_idx, base_confidence in zip(indices[0], confidences):
            pet = self.pets_database[pet_idx].copy()
            
            # Apply smart filtering and scoring adjustments
            filter_result = self._apply_filters(pet, user_answers)
            if not filter_result['passes']:
                continue  # Skip pets that don't meet hard requirements
            
            # Adjust confidence based on additional criteria
            adjusted_confidence = base_confidence * filter_result['score_multiplier']
            
            pet['match_score'] = round(adjusted_confidence, 1)
            pet['match_reason'] = self._generate_match_reason(pet, user_answers)
            
            # Clean up - remove raw features from response
            if 'raw_features' in pet:
                del pet['raw_features']
            
            recommendations.append(pet)
            
            # Stop when we have enough recommendations
            if len(recommendations) >= top_k:
                break
        
        # Assign ranks
        for rank, pet in enumerate(recommendations, 1):
            pet['rank'] = rank
        
        return recommendations
    
    def recommend_from_text(self, query_text, top_k=5, pet_type_filter=None):
        """
        Recommend pets based on natural language description
        
        Args:
            query_text: user's text description
            top_k: number of recommendations
            pet_type_filter: optional filter by pet type (e.g., "dog", "cat")
            
        Returns:
            list of pet recommendations with confidence scores
        """
        if not query_text or len(query_text.strip()) < 3:
            return []
        
        # Generate embedding for query
        query_embedding = self.sbert_model.encode([query_text])
        
        # Calculate cosine similarity with all pets
        similarities = cosine_similarity(query_embedding, self.sbert_embeddings)[0]
        
        # Apply pet type filter if specified
        if pet_type_filter:
            pet_type_filter = pet_type_filter.lower()
            filtered_indices = [i for i, pet in enumerate(self.pets_database) 
                              if pet['type'].lower() == pet_type_filter]
            if filtered_indices:
                filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
                filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                top_indices = [i for i, _ in filtered_similarities[:top_k]]
            else:
                top_indices = []
        else:
            # Get top K indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Convert to confidence scores (0-100%)
        recommendations = []
        for rank, pet_idx in enumerate(top_indices, 1):
            confidence = float(similarities[pet_idx] * 100)
            pet = self.pets_database[pet_idx].copy()
            pet['match_score'] = round(confidence, 1)
            pet['rank'] = rank
            pet['match_reason'] = f"Matches your description: \"{query_text[:60]}...\""
            
            # Clean up
            if 'raw_features' in pet:
                del pet['raw_features']
            
            recommendations.append(pet)
        
        return recommendations
    
    def get_pet_by_id(self, pet_id):
        """Get detailed information for a specific pet"""
        for pet in self.pets_database:
            if pet['id'] == pet_id:
                pet_copy = pet.copy()
                if 'raw_features' in pet_copy:
                    del pet_copy['raw_features']
                return pet_copy
        return None
    
    def get_all_pets(self, filters=None):
        """Get all pets with optional filtering"""
        pets = self.pets_database.copy()
        
        if filters:
            if 'type' in filters:
                pets = [p for p in pets if p['type'].lower() == filters['type'].lower()]
            
            if 'size' in filters:
                pets = [p for p in pets if p['size'].lower() == filters['size'].lower()]
            
            if 'kid_friendly' in filters and filters['kid_friendly']:
                pets = [p for p in pets if p['kid_friendly']]
            
            if 'energy_level' in filters:
                pets = [p for p in pets if p['energy_level'].lower() == filters['energy_level'].lower()]
        
        # Clean up raw features
        return [self._clean_pet(p) for p in pets]
    
    def _clean_pet(self, pet):
        """Remove internal fields from pet object"""
        pet_copy = pet.copy()
        if 'raw_features' in pet_copy:
            del pet_copy['raw_features']
        return pet_copy
    
    def _quiz_to_features(self, answers):
        """
        Convert quiz answers to feature vector matching training data
        
        Expected quiz answers (11 total):
        CORE FEATURES (used in ML model):
        - size_preference: 0=large, 1=medium, 2=small
        - energy_level: 0=high, 1=low, 2=moderate
        - has_kids: true/false
        - vaccinated_important: true/false
        - shedding_tolerance: 0-5 (higher = more tolerant)
        - okay_with_meat_diet: true/false
        - age_preference: 0=young (<12 months), 1=adult (12-60), 2=senior (60+)
        
        ADDITIONAL FILTERS (used in post-filtering):
        - home_type: "Apartment" | "House with a small yard" | "House with a large yard" | "Farm/Rural property"
        - current_pets: true/false
        - gender_preference: "Male" | "Female" | "No preference"
        - health_condition_acceptance: "Yes" | "No" | "Depending on condition"
        
        Feature order must match training: 
        ['Size', 'EnergyLevel', 'kid_friendliness', 'Vaccinated',
         'shedding', 'MeatConsumption', 'AgeMonths', 'WeightKg']
        """
        
        # Map answers to features
        size = answers.get('size_preference', 1)  # default medium
        energy_level = answers.get('energy_level', 2)  # default moderate
        kid_friendly = 1 if answers.get('has_kids', False) else 0
        vaccinated = 1 if answers.get('vaccinated_important', True) else 0
        shedding = answers.get('shedding_tolerance', 3)  # 0-5 range
        meat_consumption = 1 if answers.get('okay_with_meat_diet', True) else 0
        
        # Age preference to normalized age months
        age_pref = answers.get('age_preference', 1)
        if age_pref == 0:  # young
            age_months = -1.0  # normalized young age
        elif age_pref == 2:  # senior
            age_months = 2.0   # normalized senior age
        else:  # adult
            age_months = 0.0   # normalized adult age
        
        # Weight - estimate based on size preference
        weight_map = {0: 1.0, 1: 0.0, 2: -0.5}  # large, medium, small (normalized)
        weight = weight_map.get(size, 0.0)
        
        return [size, energy_level, kid_friendly, vaccinated, 
                shedding, meat_consumption, age_months, weight]
    
    def _apply_filters(self, pet, answers):
        """
        Apply smart filtering based on additional quiz questions
        
        Returns:
            dict with 'passes' (bool) and 'score_multiplier' (float 0.8-1.2)
        """
        score_multiplier = 1.0
        
        # Filter 1: Gender preference (hard filter)
        gender_pref = answers.get('gender_preference', 'No preference')
        if gender_pref != 'No preference':
            if pet['gender'].lower() != gender_pref.lower():
                return {'passes': False, 'score_multiplier': 0}
        
        # Filter 2: Home type and pet size compatibility
        home_type = answers.get('home_type', '')
        if home_type == 'Apartment':
            # Penalize large, high-energy pets in apartments
            if pet['size'] == 'Large' and pet['energy_level'] == 'High':
                score_multiplier *= 0.85
            # Bonus for small, low-energy pets
            elif pet['size'] == 'Small' and pet['energy_level'] in ['Low', 'Moderate']:
                score_multiplier *= 1.1
        
        elif home_type == 'House with a large yard':
            # Bonus for high-energy pets
            if pet['energy_level'] == 'High':
                score_multiplier *= 1.15
        
        elif home_type == 'Farm/Rural property':
            # Bonus for large, active pets
            if pet['size'] in ['Large', 'Medium'] and pet['energy_level'] == 'High':
                score_multiplier *= 1.2
        
        # Filter 3: Current pets consideration
        has_current_pets = answers.get('current_pets', False)
        if has_current_pets:
            # Prefer social, friendly pets (we can infer from kid_friendly as proxy)
            if pet['kid_friendly']:
                score_multiplier *= 1.05
        
        # Filter 4: Health condition acceptance (hard filter for special needs)
        health_acceptance = answers.get('health_condition_acceptance', 'Depending on condition')
        if health_acceptance == 'No':
            # Reject pets with poor health
            if pet['health_condition'].lower() in ['poor', 'fair']:
                return {'passes': False, 'score_multiplier': 0}
        elif health_acceptance == 'Depending on condition':
            # Accept good/excellent, penalize fair
            if pet['health_condition'].lower() == 'fair':
                score_multiplier *= 0.9
            elif pet['health_condition'].lower() == 'poor':
                return {'passes': False, 'score_multiplier': 0}
        # If "Yes", accept all health conditions
        
        # Cap multiplier to reasonable range
        score_multiplier = max(0.8, min(1.2, score_multiplier))
        
        return {'passes': True, 'score_multiplier': score_multiplier}
    
    def _generate_match_reason(self, pet, answers):
        """Generate human-readable match reason based on all 11 quiz questions"""
        reasons = []
        
        # Check kid friendliness
        if answers.get('has_kids', False) and pet['kid_friendly']:
            reasons.append("great with kids")
        
        # Check energy level
        energy_pref = answers.get('energy_level')
        if energy_pref is not None:
            energy_map = {0: "High", 1: "Low", 2: "Moderate"}
            if pet['energy_level'] == energy_map.get(energy_pref):
                reasons.append(f"matches your {pet['energy_level'].lower()} energy preference")
        
        # Check size
        size_pref = answers.get('size_preference')
        if size_pref is not None:
            size_map = {0: "Large", 1: "Medium", 2: "Small"}
            if pet['size'] == size_map.get(size_pref):
                reasons.append(f"{pet['size'].lower()} size as preferred")
        
        # Check vaccination
        if answers.get('vaccinated_important', False) and pet['vaccinated']:
            reasons.append("fully vaccinated")
        
        # Check age
        age_pref = answers.get('age_preference')
        if age_pref == 0 and pet['age_months'] < 12:
            reasons.append("young and energetic")
        elif age_pref == 2 and pet['age_months'] > 60:
            reasons.append("mature and calm")
        elif age_pref == 1 and 12 <= pet['age_months'] <= 60:
            reasons.append("perfect adult age")
        
        # Check home type compatibility
        home_type = answers.get('home_type', '')
        if home_type == 'Apartment' and pet['size'] == 'Small':
            reasons.append("apartment-friendly size")
        elif home_type in ['House with a large yard', 'Farm/Rural property'] and pet['energy_level'] == 'High':
            reasons.append("loves outdoor space")
        
        # Check gender match
        gender_pref = answers.get('gender_preference', 'No preference')
        if gender_pref != 'No preference' and pet['gender'].lower() == gender_pref.lower():
            reasons.append(f"{pet['gender'].lower()} as requested")
        
        # Check health condition
        if pet['health_condition'] == 'Excellent':
            reasons.append("excellent health")
        
        # Check if good with other pets (using kid_friendly as proxy for sociability)
        if answers.get('current_pets', False) and pet['kid_friendly']:
            reasons.append("friendly with other pets")
        
        if not reasons:
            reasons.append("good overall match based on your preferences")
        
        # Return top 3-4 most relevant reasons
        return "Perfect match: " + ", ".join(reasons[:4]) + "!"
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {
            'total_pets': len(self.pets_database),
            'by_type': {},
            'by_size': {},
            'by_energy': {},
            'vaccinated_count': sum(1 for p in self.pets_database if p['vaccinated']),
            'kid_friendly_count': sum(1 for p in self.pets_database if p['kid_friendly'])
        }
        
        for pet in self.pets_database:
            # Count by type
            pet_type = pet['type']
            stats['by_type'][pet_type] = stats['by_type'].get(pet_type, 0) + 1
            
            # Count by size
            size = pet['size']
            stats['by_size'][size] = stats['by_size'].get(size, 0) + 1
            
            # Count by energy
            energy = pet['energy_level']
            stats['by_energy'][energy] = stats['by_energy'].get(energy, 0) + 1
        
        return stats
