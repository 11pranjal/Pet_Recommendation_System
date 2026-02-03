"""
Pet Recommendation Engine
- KNN-based quiz recommendations with confidence scores
- SBERT-based text search with fair hybrid search (keyword + semantic)
- Returns pet names and breeds (not numbers!)
"""

import numpy as np
import joblib
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
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
            user_answers: dict with quiz answers
            top_k: number of recommendations
            
        Returns:
            list of pet recommendations with confidence scores
        """
        # Convert user answers to feature vector
        feature_vector = self._quiz_to_features(user_answers)
        
        # Scale features
        feature_scaled = self.knn_scaler.transform([feature_vector])
        
        # Get KNN predictions with distances
        distances, indices = self.knn_model.kneighbors(feature_scaled, n_neighbors=min(top_k, len(self.pets_database)))
        
        # Convert distances to confidence percentage (0-100%)
        # Smaller distance = higher confidence
        if distances[0].max() > 0:
            max_dist = distances[0].max()
            # Normalize: confidence = (1 - normalized_distance) * 100
            confidences = [(1 - (dist / max_dist)) * 100 for dist in distances[0]]
        else:
            confidences = [100.0] * len(distances[0])
        
        # Get pet recommendations
        recommendations = []
        for rank, (pet_idx, confidence) in enumerate(zip(indices[0], confidences), 1):
            pet = self.pets_database[pet_idx].copy()
            pet['match_score'] = round(confidence, 1)
            pet['rank'] = rank
            pet['match_reason'] = self._generate_match_reason(pet, user_answers)
            
            # Clean up - remove raw features from response
            if 'raw_features' in pet:
                del pet['raw_features']
            
            recommendations.append(pet)
        
        return recommendations
    
    def recommend_from_text(self, query_text, top_k=5, pet_type_filter=None):
        """
        Recommend pets based on natural language description with fair hybrid search
        Combines semantic similarity with keyword matching for better diversity
        
        Args:
            query_text: user's text description
            top_k: number of recommendations
            pet_type_filter: optional filter by pet type (e.g., "dog", "cat")
            
        Returns:
            list of pet recommendations with confidence scores
        """
        if not query_text or len(query_text.strip()) < 3:
            return []
        
        # Check if user specified a specific pet type or breed in the query
        query_lower = query_text.lower()
        user_specified_type = None
        user_specified_breed = None
        
        # Check for generic pet types
        for pet_type in ['dog', 'cat', 'bird', 'rabbit']:
            if pet_type in query_lower:
                user_specified_type = pet_type.capitalize()
                break
        
        # Check for specific breed names
        all_breeds = set(pet['breed'] for pet in self.pets_database)
        for breed in all_breeds:
            if breed.lower() in query_lower:
                user_specified_breed = breed
                # Also infer the pet type from the breed
                for pet in self.pets_database:
                    if pet['breed'] == breed:
                        user_specified_type = pet['type']
                        break
                break
        
        # If user specified a type, use it as filter
        if user_specified_type:
            pet_type_filter = user_specified_type
        
        # Extract keywords for physical attributes
        keywords = self._extract_keywords(query_text)
        
        # Generate embedding for query
        query_embedding = self.sbert_model.encode([query_text])
        
        # Calculate cosine similarity with all pets
        semantic_scores = cosine_similarity(query_embedding, self.sbert_embeddings)[0]
        
        # Calculate hybrid scores with keyword matching
        results = []
        for i, pet in enumerate(self.pets_database):
            semantic_score = semantic_scores[i]
            
            # Skip if filtered by pet type
            if pet_type_filter and pet['type'].lower() != pet_type_filter.lower():
                continue
            
            # Skip if filtered by breed
            if user_specified_breed and pet.get('breed', '').lower() != user_specified_breed.lower():
                continue
            
            # Calculate keyword match score
            if keywords:
                # Build text from available fields
                text_parts = []
                
                # Use pet_details and pet_characteristics if available (SBERT dataset)
                if 'pet_details' in pet or 'pet_characteristics' in pet:
                    if 'pet_details' in pet:
                        text_parts.append(pet['pet_details'])
                    if 'pet_characteristics' in pet:
                        text_parts.append(pet['pet_characteristics'])
                
                # Fallback to description if pet_details not available (KNN dataset)
                if not text_parts and 'description' in pet:
                    text_parts.append(pet['description'])
                
                # Add size, type and breed info (important for size matching)
                text_parts.append(f"{pet['type']} {pet.get('breed', '')} {pet.get('size', '').lower()}")
                
                text = " ".join(text_parts)
                keyword_score = self._keyword_match_score(text, keywords)
                
                # BOOST: If size keyword is in query and Size field matches exactly, add bonus
                size_keywords = ['small', 'medium', 'large']
                has_size_keyword = any(sk in keywords for sk in size_keywords)
                size_field_match = False
                
                for size_kw in size_keywords:
                    if size_kw in keywords and pet.get('size', '').lower() == size_kw:
                        size_field_match = True
                        break
                
                # If perfect keyword match, apply diversity-aware ranking
                # BUT ONLY if user didn't specify a pet type
                perfect_match = keyword_score == 1.0
                
                # Adaptive scoring based on whether size is mentioned
                if has_size_keyword:
                    # When size is mentioned, prioritize size field match heavily
                    if size_field_match:
                        # Size field matches: 5% semantic (just for tiebreaking), 95% size bonus
                        # This ensures all pets with correct size field rank similarly, regardless of breed semantic similarity
                        final_score = 0.05 * semantic_score + 0.95
                    else:
                        # Size field doesn't match: heavily penalize
                        final_score = 0.3 * semantic_score + 0.2 * keyword_score
                else:
                    # No size keyword: standard hybrid scoring
                    final_score = 0.5 * semantic_score + 0.5 * keyword_score
            else:
                keyword_score = 0
                perfect_match = False
                final_score = semantic_score
            
            results.append({
                'idx': i,
                'pet': pet,
                'semantic': semantic_score,
                'keyword': keyword_score,
                'final': final_score,
                'perfect_match': perfect_match
            })
        
        # Apply diversity-aware ranking ONLY if:
        # 1. We have perfect matches
        # 2. User did NOT specify a pet type
        # 3. No pet_type_filter was applied
        # 4. User did NOT specify a specific breed
        # OR if size keyword is present and size field matches (ensure breed diversity for same-sized pets)
        perfect_matches = [r for r in results if r['perfect_match']]
        
        # Check if we have size-based filtering (all results with similar scores due to size bonus)
        has_size_filter = any(sk in keywords for sk in ['small', 'medium', 'large']) if keywords else False
        size_filtered_results = [r for r in results if r['final'] > 0.9] if has_size_filter else []
        
        if perfect_matches and len(perfect_matches) > top_k and not pet_type_filter and not user_specified_type and not user_specified_breed:
            # Group by pet type for diversity
            by_type = defaultdict(list)
            for r in perfect_matches:
                by_type[r['pet']['type']].append(r)
            
            # Sort within each type by semantic score
            for pet_type in by_type:
                by_type[pet_type].sort(key=lambda x: x['semantic'], reverse=True)
            
            # Round-robin selection for diversity
            diverse_results = []
            type_indices = {t: 0 for t in by_type}
            types_list = list(by_type.keys())
            
            while len(diverse_results) < top_k and any(type_indices[t] < len(by_type[t]) for t in types_list):
                for pet_type in types_list:
                    if type_indices[pet_type] < len(by_type[pet_type]):
                        diverse_results.append(by_type[pet_type][type_indices[pet_type]])
                        type_indices[pet_type] += 1
                        
                        if len(diverse_results) >= top_k:
                            break
            
            top_results = diverse_results[:top_k]
        elif has_size_filter and len(size_filtered_results) > top_k and pet_type_filter and not user_specified_breed:
            # Size-based filtering with pet type specified BUT no specific breed: ensure breed diversity
            by_breed = defaultdict(list)
            for r in size_filtered_results:
                by_breed[r['pet'].get('breed', 'Unknown')].append(r)
            
            # Sort within each breed by semantic score
            for breed in by_breed:
                by_breed[breed].sort(key=lambda x: x['semantic'], reverse=True)
            
            # Round-robin selection for breed diversity
            diverse_results = []
            breed_indices = {b: 0 for b in by_breed}
            breeds_list = list(by_breed.keys())
            
            while len(diverse_results) < top_k and any(breed_indices[b] < len(by_breed[b]) for b in breeds_list):
                for breed in breeds_list:
                    if breed_indices[breed] < len(by_breed[breed]):
                        diverse_results.append(by_breed[breed][breed_indices[breed]])
                        breed_indices[breed] += 1
                        
                        if len(diverse_results) >= top_k:
                            break
            
            top_results = diverse_results[:top_k]
        else:
            # Normal ranking by final score (including when specific breed is mentioned)
            results.sort(key=lambda x: x['final'], reverse=True)
            top_results = results[:top_k]
        
        # Convert to recommendations
        recommendations = []
        for rank, result in enumerate(top_results, 1):
            # Convert final score to confidence (0-100%)
            confidence = float(result['final'] * 100)
            
            pet = result['pet'].copy()
            pet['match_score'] = round(confidence, 1)
            pet['rank'] = rank
            
            # Generate match reason
            if keywords and result['keyword'] > 0:
                pet['match_reason'] = f"Matches your description with keywords: {', '.join(keywords[:2])}"
            else:
                pet['match_reason'] = f"Semantic match: \"{query_text[:50]}...\""
            
            # Clean up
            if 'raw_features' in pet:
                del pet['raw_features']
            
            recommendations.append(pet)
        
        return recommendations
    
    def _extract_keywords(self, query):
        """Extract physical attribute keywords from query"""
        physical_keywords = [
            'two feet', 'two legs', 'bipedal',
            'four legs', 'four paws', 'quadruped',
            'wings', 'feathers', 'beak',
            'tail', 'whiskers', 'fur', 'scales', 'claws', 'hooves', 'ears'
        ]
        
        # Size keywords for diversity
        size_keywords = ['small', 'medium', 'large', 'tiny', 'big', 'compact']
        
        query_lower = query.lower()
        
        # Normalize numbers to words
        query_lower = query_lower.replace('2', 'two')
        query_lower = query_lower.replace('4', 'four')
        
        # Fix singular/plural variations - need to handle all positions
        # Replace singular with plural for consistency
        import re
        query_lower = re.sub(r'\bleg\b', 'legs', query_lower)
        query_lower = re.sub(r'\bfoot\b', 'feet', query_lower)
        query_lower = re.sub(r'\bpaw\b', 'paws', query_lower)
        query_lower = re.sub(r'\bwing\b', 'wings', query_lower)
        
        # Fix common grammar errors
        query_lower = query_lower.replace('feets', 'feet')
        query_lower = query_lower.replace('foots', 'feet')
        query_lower = query_lower.replace('legss', 'legs')
        query_lower = query_lower.replace('pawss', 'paws')
        
        found_keywords = []
        
        # Check physical keywords
        for keyword in physical_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        # Check size keywords
        for keyword in size_keywords:
            # Use word boundaries to avoid partial matches
            if f" {keyword} " in f" {query_lower} " or query_lower.startswith(keyword) or query_lower.endswith(keyword):
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _keyword_match_score(self, text, keywords):
        """Calculate keyword match score (0.0 to 1.0)"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for kw in keywords:
            # Check in text for physical keywords
            if kw in text_lower:
                matches += 1
            # Special case: "two legs" and "two feet" are equivalent (birds have both)
            elif kw == 'two legs' and 'two feet' in text_lower:
                matches += 1
            elif kw == 'two feet' and 'two legs' in text_lower:
                matches += 1
            # For size keywords, also check the 'size' field if available
            elif kw in ['small', 'medium', 'large'] and kw in text_lower:
                matches += 1
        
        return matches / len(keywords)
    
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
        """Remove internal fields from pet object and normalize format"""
        pet_copy = pet.copy()
        if 'raw_features' in pet_copy:
            del pet_copy['raw_features']
        
        # Normalize pet_id field - add it if only 'id' exists
        if 'id' in pet_copy and 'pet_id' not in pet_copy:
            pet_copy['pet_id'] = pet_copy['id']
        
        # Calculate age_years from age_months if not present
        if 'age_months' in pet_copy and 'age_years' not in pet_copy:
            total_months = pet_copy['age_months']
            pet_copy['age_years'] = total_months // 12
            pet_copy['age_months'] = total_months % 12
        
        return pet_copy
    
    def _quiz_to_features(self, answers):
        """
        Convert quiz answers to feature vector matching training data
        
        Expected quiz answers:
        - pet_type: 0=bird, 1=cat, 2=dog, 3=rabbit (optional filter)
        - size_preference: 0=large, 1=medium, 2=small
        - energy_level: 0=high, 1=low, 2=moderate
        - has_kids: true/false
        - vaccinated_important: true/false
        - shedding_tolerance: 0-5 (higher = more tolerant)
        - okay_with_meat_diet: true/false
        - age_preference: 0=young (<12 months), 1=adult (12-60), 2=senior (60+)
        
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
    
    def _generate_match_reason(self, pet, answers):
        """Generate human-readable match reason"""
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
        
        if not reasons:
            reasons.append("good overall match based on your preferences")
        
        return "Perfect match: " + ", ".join(reasons[:3]) + "!"
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {
            'total_pets': len(self.pets_database),
            'by_type': {},
            'by_size': {},
        }
        
        # Check if database has these fields (backwards compatibility)
        if self.pets_database and len(self.pets_database) > 0:
            sample_pet = self.pets_database[0]
            
            if 'vaccinated' in sample_pet:
                stats['vaccinated_count'] = sum(1 for p in self.pets_database if p.get('vaccinated', False))
            
            if 'kid_friendly' in sample_pet:
                stats['kid_friendly_count'] = sum(1 for p in self.pets_database if p.get('kid_friendly', False))
            
            if 'energy_level' in sample_pet:
                stats['by_energy'] = {}
                for pet in self.pets_database:
                    energy = pet.get('energy_level', 'Unknown')
                    stats['by_energy'][energy] = stats['by_energy'].get(energy, 0) + 1
        
        for pet in self.pets_database:
            # Count by type
            pet_type = pet['type']
            stats['by_type'][pet_type] = stats['by_type'].get(pet_type, 0) + 1
            
            # Count by size
            size = pet.get('size', 'Unknown')
            stats['by_size'][size] = stats['by_size'].get(size, 0) + 1
        
        return stats
