"""
Pet Recommendation Engine
- KNN-based quiz recommendations with confidence scores
- SBERT-based text search with fair hybrid search (keyword + semantic)
- Returns pet names and breeds (not numbers!)
"""

import numpy as np
import pickle
from datetime import date
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

class PetRecommendationEngine:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.pets_database = None
        self.knn_X = None  # feature matrix for nearest-neighbor distance calc
        self.sbert_model = None
        self.sbert_embeddings = None
        self.sbert_desc_embeddings = None  # pre-computed description embeddings
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
        
        # Load KNN feature matrix (new: raw features; fallback: old double-scaled)
        knn_x_path = os.path.join(self.model_dir, "knn_X.npy")
        knn_x_old_path = os.path.join(self.model_dir, "knn_X_scaled.npy")
        if os.path.exists(knn_x_path):
            self.knn_X = np.load(knn_x_path)
            print("✓ Loaded KNN feature matrix")
        elif os.path.exists(knn_x_old_path):
            self.knn_X = np.load(knn_x_old_path)
            print("⚠️  Loaded old knn_X_scaled.npy — retrain with enhanced_training.py for best results")
        else:
            raise FileNotFoundError("KNN feature matrix not found! Please run training first.")
        
        # Load feature columns
        with open(os.path.join(self.model_dir, "knn_features.txt"), "r") as f:
            self.feature_columns = f.read().strip().split(",")
        print(f"✓ KNN features: {self.feature_columns}")
        
        # Load SBERT model and embeddings
        self.sbert_model = SentenceTransformer(os.path.join(self.model_dir, "sbert_model"))
        self.sbert_embeddings = np.load(os.path.join(self.model_dir, "sbert_embeddings.npy"))
        print("✓ Loaded SBERT model and embeddings")

        # Load pre-computed description embeddings (optional — falls back to per-query encoding)
        desc_emb_path = os.path.join(self.model_dir, "sbert_desc_embeddings.npy")
        if os.path.exists(desc_emb_path):
            self.sbert_desc_embeddings = np.load(desc_emb_path)
            print("✓ Loaded pre-computed description embeddings")
        else:
            self.sbert_desc_embeddings = None
            print("⚠️  No pre-computed description embeddings — will encode per query")
        
        print("✅ All models loaded successfully!\n")
    
    def recommend_from_quiz(self, user_answers, top_k=5):
        """
        Recommend pets based on quiz answers with hard-constraint filtering
        and type-diverse results.
        
        Strategy:
          1. Filter by hard constraints (meat, pet_type, gender)
          2. Rank all candidates by euclidean distance
          3. Apply type-diversity: guarantee the best match from each
             eligible pet type appears in the results, then fill remaining
             slots by overall distance ranking.
        
        This prevents the dataset's distribution skew from causing
        results to be dominated by a single pet type.
        """
        # Convert user answers to feature vector
        # Values are already in the same space as the CSV (integer codes + Z-scores)
        query_features = np.array(self._quiz_to_features(user_answers), dtype=float)
        
        # --- Build hard-constraint filters ---
        hard_filters = {}
        
        # Meat diet preference: 'yes' = only meat pets, 'no' = only herbivores, 'any' = no filter
        meat_pref = user_answers.get('meat_diet_preference', 'any')
        if meat_pref == 'yes':
            hard_filters['meat_consumption'] = True
        elif meat_pref == 'no':
            hard_filters['meat_consumption'] = False
        
        # Pet type: if user specified a type, only show that type
        pet_type_pref = user_answers.get('pet_type')
        if pet_type_pref is not None:
            type_map = {0: 'Bird', 1: 'Cat', 2: 'Dog', 3: 'Rabbit'}
            hard_filters['type'] = type_map.get(pet_type_pref)
        
        # Gender preference
        gender_pref = user_answers.get('gender_preference')
        if gender_pref and gender_pref != 'No preference':
            hard_filters['gender'] = gender_pref
        
        # --- Filter eligible pets, then rank by feature distance ---
        candidates = []
        knn_size = len(self.knn_X) if self.knn_X is not None else 0
        for idx, pet in enumerate(self.pets_database):
            # Apply hard filters
            if 'meat_consumption' in hard_filters:
                if pet.get('meat_consumption') != hard_filters['meat_consumption']:
                    continue
            if 'type' in hard_filters:
                if pet.get('type') != hard_filters['type']:
                    continue
            if 'gender' in hard_filters:
                if pet.get('gender') != hard_filters['gender']:
                    continue

            # Safety: skip if KNN features not available for this index
            # (can happen if custom pet's raw_features were empty)
            if idx >= knn_size:
                continue

            # Compute euclidean distance in feature space
            pet_features = self.knn_X[idx]
            dist = np.sqrt(np.sum((query_features - pet_features) ** 2))
            candidates.append((idx, dist))
        
        # Sort by distance (closest = best match)
        candidates.sort(key=lambda x: x[1])
        
        # --- Type-diverse selection ---
        # If user didn't filter by a specific type, ensure diversity
        if 'type' not in hard_filters and top_k >= 3:
            # Find the best candidate of each pet type
            best_by_type = {}  # type -> (idx, dist)
            for (pet_idx, dist) in candidates:
                ptype = self.pets_database[pet_idx].get('type', '')
                if ptype not in best_by_type:
                    best_by_type[ptype] = (pet_idx, dist)
            
            # Build diverse result: first include the best from each type
            diverse_picks = []          # list of (pet_idx, dist)
            picked_indices = set()
            
            # Sort types by their best distance (best-matching type first)
            sorted_types = sorted(best_by_type.items(), key=lambda x: x[1][1])
            for ptype, (pet_idx, dist) in sorted_types:
                diverse_picks.append((pet_idx, dist))
                picked_indices.add(pet_idx)
            
            # Fill remaining slots from overall ranking
            for (pet_idx, dist) in candidates:
                if len(diverse_picks) >= top_k:
                    break
                if pet_idx not in picked_indices:
                    diverse_picks.append((pet_idx, dist))
                    picked_indices.add(pet_idx)
            
            # Re-sort the final picks by distance
            diverse_picks.sort(key=lambda x: x[1])
            final_candidates = diverse_picks[:top_k]
        else:
            final_candidates = candidates[:top_k]
        
        # Convert to recommendations
        scale_factor = 0.15
        recommendations = []
        for rank, (pet_idx, dist) in enumerate(final_candidates, 1):
            pet = self._clean_pet(self.pets_database[pet_idx])

            confidence = 100.0 / (1.0 + dist * scale_factor)
            pet['match_score'] = round(confidence, 1)
            pet['rank'] = rank
            pet['match_reason'] = self._generate_match_reason(pet, user_answers)

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
        
        # Extract structured attribute preferences from query
        attribute_prefs = self._extract_attribute_preferences(query_text)
        
        # Generate embedding for query
        query_embedding = self.sbert_model.encode([query_text])
        
        # Calculate cosine similarity with all pets
        semantic_scores = cosine_similarity(query_embedding, self.sbert_embeddings)[0]
        
        # --- Description-specific similarity for top candidates ---
        top_candidate_indices = np.argsort(semantic_scores)[::-1][:50].tolist()
        desc_scores = {}
        if self.sbert_desc_embeddings is not None:
            # Use pre-computed description embeddings (fast path)
            valid_indices = [i for i in top_candidate_indices if i < len(self.sbert_desc_embeddings)]
            if valid_indices:
                desc_emb_batch = self.sbert_desc_embeddings[valid_indices]
                desc_sims = cosine_similarity(query_embedding, desc_emb_batch)[0]
                for pos, idx in enumerate(valid_indices):
                    desc_scores[idx] = float(desc_sims[pos])
        else:
            # Fallback: encode descriptions per query (slow path)
            desc_texts = []
            desc_idx_map = []
            for idx in top_candidate_indices:
                if idx < len(self.pets_database):
                    desc = self.pets_database[idx].get('description', '') or ''
                    if desc and len(desc.strip()) > 5:
                        desc_texts.append(desc)
                        desc_idx_map.append(idx)
            if desc_texts:
                desc_embeddings = self.sbert_model.encode(desc_texts)
                desc_sims = cosine_similarity(query_embedding, desc_embeddings)[0]
                for pos, idx in enumerate(desc_idx_map):
                    desc_scores[idx] = float(desc_sims[pos])
        
        # --- Determine which attributes should act as HARD FILTERS ---
        # When user explicitly says "small black pet", they do NOT want large or gray pets.
        # These attributes are clear-cut: size, color, meat_consumption, vaccinated, kid_friendly
        hard_attribute_filters = {}
        soft_attribute_prefs = {}
        HARD_FILTER_ATTRS = {'size', 'color', 'meat_consumption', 'vaccinated', 'kid_friendly'}
        for attr_name, attr_value in attribute_prefs.items():
            if attr_name in HARD_FILTER_ATTRS:
                hard_attribute_filters[attr_name] = attr_value
            else:
                soft_attribute_prefs[attr_name] = attr_value
        
        # Calculate hybrid scores with keyword matching + structured attribute boosting
        results = []
        for i, pet in enumerate(self.pets_database):
            # Bounds check: skip if embeddings are out of sync
            if i >= len(semantic_scores):
                break
            
            semantic_score = semantic_scores[i]
            description_score = desc_scores.get(i, 0.0)
            
            # Skip if filtered by pet type
            if pet_type_filter and pet['type'].lower() != pet_type_filter.lower():
                continue
            
            # Skip if filtered by breed
            if user_specified_breed and pet.get('breed', '').lower() != user_specified_breed.lower():
                continue
            
            # --- HARD FILTER: skip pets that violate explicit attribute requests ---
            # e.g. user says "small black" → exclude non-Small and non-black pets entirely
            skip_pet = False
            for attr_name, attr_value in hard_attribute_filters.items():
                if attr_name == 'size':
                    if pet.get('size', '').lower() != attr_value.lower():
                        skip_pet = True
                        break
                elif attr_name == 'color':
                    # attr_value is a list of colors, e.g. ['black'] or ['black', 'white']
                    # Pet matches if ANY user color appears in the pet's color string
                    # Uses word-boundary matching to avoid false positives
                    # (e.g. "red" should NOT match "cream")
                    import re
                    pet_color = pet.get('color', '').lower()
                    color_match = False
                    for c in attr_value:
                        if re.search(r'\b' + re.escape(c) + r'\b', pet_color):
                            color_match = True
                            break
                        # grey ↔ gray equivalence
                        alt = 'grey' if c == 'gray' else ('gray' if c == 'grey' else None)
                        if alt and re.search(r'\b' + re.escape(alt) + r'\b', pet_color):
                            color_match = True
                            break
                    if not color_match:
                        skip_pet = True
                        break
                elif attr_name == 'meat_consumption':
                    if pet.get('meat_consumption') != attr_value:
                        skip_pet = True
                        break
                elif attr_name == 'vaccinated':
                    if pet.get('vaccinated') != attr_value:
                        skip_pet = True
                        break
                elif attr_name == 'kid_friendly':
                    if pet.get('kid_friendly') != attr_value:
                        skip_pet = True
                        break
            if skip_pet:
                continue
            
            # --- Soft attribute boost (for remaining non-hard attributes) ---
            attr_boost = 0.0
            attr_count = 0
            if soft_attribute_prefs:
                for attr_name, attr_value in soft_attribute_prefs.items():
                    attr_count += 1
                    if attr_name == 'energy_level':
                        if pet.get('energy_level', '').lower() == attr_value.lower():
                            attr_boost += 1.0
                        else:
                            attr_boost -= 0.5
                    elif attr_name == 'health_condition':
                        if pet.get('health_condition', '').lower() == attr_value.lower():
                            attr_boost += 1.0
                    elif attr_name == 'has_previous_owner':
                        if pet.get('has_previous_owner') == attr_value:
                            attr_boost += 0.5
                    elif attr_name == 'shedding_level':
                        shed = pet.get('shedding_level', 3)
                        if attr_value == 'low' and shed <= 1:
                            attr_boost += 1.0
                        elif attr_value == 'low' and shed <= 2:
                            attr_boost += 0.5
                        elif attr_value == 'high' and shed >= 4:
                            attr_boost += 1.0
                        elif attr_value == 'high' and shed >= 3:
                            attr_boost += 0.5
                        else:
                            attr_boost -= 0.3
                
                # Normalize boost to 0-1 range
                if attr_count > 0:
                    attr_boost_normalized = (attr_boost / attr_count + 0.5) / 1.5
                    attr_boost_normalized = max(0.0, min(1.0, attr_boost_normalized))
                else:
                    attr_boost_normalized = 0.5
            else:
                attr_boost_normalized = 0.5  # neutral when no soft prefs
            
            # Calculate keyword match score
            if keywords:
                # Build text from available fields (filter None/empty to avoid crashes)
                text_parts = []
                
                # Use pet_details and pet_characteristics if available (SBERT dataset)
                if pet.get('pet_details'):
                    text_parts.append(str(pet['pet_details']))
                if pet.get('pet_characteristics'):
                    text_parts.append(str(pet['pet_characteristics']))
                
                # Fallback to description if neither pet_details nor pet_characteristics had content
                if not text_parts and pet.get('description'):
                    text_parts.append(str(pet['description']))
                
                # Add structured fields for keyword matching: type, breed, size, AND color
                text_parts.append(
                    f"{pet['type']} {pet.get('breed', '')} "
                    f"{pet.get('size', '').lower()} {pet.get('color', '').lower()}"
                )
                
                text = " ".join(text_parts)
                keyword_score = self._keyword_match_score(text, keywords)
                
                # If perfect keyword match, apply diversity-aware ranking
                perfect_match = keyword_score == 1.0
                
                # --- Scoring formula adapts based on what the user asked for ---
                if attribute_prefs:
                    # User asked for specific attributes → attributes already hard-filtered above,
                    # so among remaining candidates, semantic + keyword + description matter most
                    if soft_attribute_prefs:
                        final_score = 0.35 * semantic_score + 0.25 * keyword_score + 0.20 * attr_boost_normalized + 0.20 * description_score
                    else:
                        # Only hard attrs were requested (e.g. just "small pet")
                        # All remaining pets already match the attribute — rank by semantic + keyword
                        final_score = 0.40 * semantic_score + 0.35 * keyword_score + 0.25 * description_score
                else:
                    # No structured attributes detected — pure semantic + keyword search
                    final_score = 0.40 * semantic_score + 0.40 * keyword_score + 0.20 * description_score
            else:
                keyword_score = 0
                perfect_match = False
                # No physical keywords found
                if soft_attribute_prefs:
                    final_score = 0.40 * semantic_score + 0.30 * attr_boost_normalized + 0.30 * description_score
                elif hard_attribute_filters:
                    # Only hard filters (already applied above) — rank by semantic + description
                    final_score = 0.55 * semantic_score + 0.45 * description_score
                else:
                    # Pure semantic search (no attributes at all)
                    final_score = 0.65 * semantic_score + 0.35 * description_score
            
            results.append({
                'idx': i,
                'pet': pet,
                'semantic': semantic_score,
                'keyword': keyword_score,
                'final': final_score,
                'perfect_match': perfect_match
            })
        
        # --- Impossible combination check ---
        # If physical keywords were extracted (e.g. "wings" from "flying dog")
        # but EVERY candidate scored 0 on keywords, the query describes something
        # that doesn't exist (e.g. a flying dog). Return empty instead of noise.
        if keywords and results:
            best_keyword_score = max(r['keyword'] for r in results)
            if best_keyword_score == 0:
                return []
        
        # --- Diversity-aware ranking ---
        # When many results have similar scores, diversify by type or breed
        perfect_matches = [r for r in results if r['perfect_match']]
        has_size_filter = 'size' in hard_attribute_filters
        
        if perfect_matches and len(perfect_matches) > top_k and not pet_type_filter and not user_specified_type and not user_specified_breed:
            # Group by pet type for diversity
            by_type = defaultdict(list)
            for r in perfect_matches:
                by_type[r['pet']['type']].append(r)
            
            # Sort within each type by final score
            for pet_type in by_type:
                by_type[pet_type].sort(key=lambda x: x['final'], reverse=True)
            
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
        elif has_size_filter and len(results) > top_k and not user_specified_breed:
            # Size was hard-filtered, so all results are the right size.
            # Ensure breed diversity among same-sized pets
            by_breed = defaultdict(list)
            for r in results:
                by_breed[r['pet'].get('breed', 'Unknown')].append(r)
            
            # Sort within each breed by final score
            for breed in by_breed:
                by_breed[breed].sort(key=lambda x: x['final'], reverse=True)
            
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
            # Normal ranking by final score
            results.sort(key=lambda x: x['final'], reverse=True)
            top_results = results[:top_k]
        
        # Convert to recommendations
        recommendations = []
        for rank, result in enumerate(top_results, 1):
            # Convert final score to confidence (0-100%)
            confidence = float(result['final'] * 100)

            pet = self._clean_pet(result['pet'])
            pet['match_score'] = round(confidence, 1)
            pet['rank'] = rank

            # Generate match reason
            reason_parts = []
            if hard_attribute_filters:
                matched_attrs = []
                for k, v in hard_attribute_filters.items():
                    if isinstance(v, list):
                        matched_attrs.append(f"{k}: {', '.join(str(x) for x in v)}")
                    else:
                        matched_attrs.append(f"{k}: {v}")
                reason_parts.append(f"Matches: {', '.join(matched_attrs)}")
            if keywords and result['keyword'] > 0:
                reason_parts.append(f"Keywords: {', '.join(keywords[:3])}")
            if not reason_parts:
                reason_parts.append(f"Semantic match: \"{query_text[:50]}\"")
            pet['match_reason'] = "; ".join(reason_parts)

            recommendations.append(pet)
        
        return recommendations
    
    def _extract_attribute_preferences(self, query):
        """Extract structured attribute preferences from natural language query.
        Maps user phrases to actual pet database fields for precise filtering/boosting.
        
        For conflicting inputs (e.g. "big big small"), uses the LAST mentioned
        word as the user's final intent.
        """
        import re
        q = query.lower().strip()
        prefs = {}
        
        # --- Energy level ---
        low_energy_patterns = [
            r'\blow[\s-]?energy\b', r'\bcalm\b', r'\brelaxed\b', r'\blazy\b',
            r'\bquiet\b', r'\bchill\b', r'\blaid[\s-]?back\b', r'\bcouch potato\b',
            r'\bnot (very )?active\b', r'\bless active\b', r'\beasygoing\b',
            r'\blow[\s-]?maintenance\b'
        ]
        high_energy_patterns = [
            r'\bhigh[\s-]?energy\b', r'\bactive\b', r'\benergetic\b', r'\bplayful\b',
            r'\bhyper\b', r'\bvery active\b', r'\bsporty\b', r'\bathletic\b',
            r'\brun\b', r'\brunning\b', r'\bjog\b', r'\bjogging\b'
        ]
        moderate_energy_patterns = [
            r'\bmoderate[\s-]?energy\b', r'\bmedium[\s-]?energy\b', r'\bnormal energy\b'
        ]
        
        if any(re.search(p, q) for p in low_energy_patterns):
            prefs['energy_level'] = 'Low'
        elif any(re.search(p, q) for p in moderate_energy_patterns):
            prefs['energy_level'] = 'Moderate'
        elif any(re.search(p, q) for p in high_energy_patterns):
            prefs['energy_level'] = 'High'
        
        # --- Vaccination ---
        if re.search(r'\bvaccinat(ed|ion)\b', q):
            if re.search(r'\bnot\s+vaccinat', q) or re.search(r'\bunvaccinat', q):
                prefs['vaccinated'] = False
            else:
                prefs['vaccinated'] = True
        
        # --- Kid friendly ---
        kid_friendly_patterns = [
            r'\bkid[\s-]?friendly\b', r'\bchild[\s-]?friendly\b', r'\bfamily[\s-]?friendly\b',
            r'\bgood with (kids|children|family)\b', r'\bgreat with (kids|children|family)\b',
            r'\bsafe (for|with) (kids|children)\b', r'\bfamily pet\b',
            r'\bhave (kids|children)\b', r'\bwith (kids|children)\b'
        ]
        if any(re.search(p, q) for p in kid_friendly_patterns):
            prefs['kid_friendly'] = True
        
        # --- Meat consumption ---
        non_meat_patterns = [
            r'\bno meat\b', r'\bnon[\s-]?meat\b', r'\bvegetarian\b', r'\bvegan\b',
            r'\bherbivore\b', r'\bplant[\s-]?based\b', r'\bno[\s-]?meat[\s-]?diet\b',
            r'\bdoesn.t eat meat\b', r'\bwithout meat\b'
        ]
        meat_patterns = [
            r'\bmeat[\s-]?based\b', r'\bmeat[\s-]?eater\b', r'\bcarnivore\b',
            r'\beats meat\b'
        ]
        if any(re.search(p, q) for p in non_meat_patterns):
            prefs['meat_consumption'] = False
        elif any(re.search(p, q) for p in meat_patterns):
            prefs['meat_consumption'] = True
        
        # --- Health ---
        if re.search(r'\b(excellent|perfect|great) health\b', q):
            prefs['health_condition'] = 'Excellent'
        elif re.search(r'\bhealthy\b', q) or re.search(r'\bgood health\b', q):
            prefs['health_condition'] = 'Excellent'
        
        # --- Shedding ---
        if re.search(r'\b(low|no|minimal|less|little)[\s-]?shedd?ing\b', q) or re.search(r'\bhypoallergenic\b', q):
            prefs['shedding_level'] = 'low'
        elif re.search(r'\b(heavy|high|lots? of)[\s-]?shedd?ing\b', q):
            prefs['shedding_level'] = 'high'
        
        # --- Size (LAST-WINS: scan left-to-right, the last size word is the user's final intent) ---
        # This correctly handles "big big small" → Small, "small large" → Large, etc.
        size_patterns = [
            (r'\bsmall\b', 'Small'), (r'\btiny\b', 'Small'), (r'\bcompact\b', 'Small'),
            (r'\bmedium\b', 'Medium'),
            (r'\blarge\b', 'Large'), (r'\bbig\b', 'Large'),
        ]
        last_size = None
        last_size_pos = -1
        for pattern, size_val in size_patterns:
            for m in re.finditer(pattern, q):
                if m.start() > last_size_pos:
                    last_size_pos = m.start()
                    last_size = size_val
        if last_size:
            prefs['size'] = last_size
        
        # --- Color (all mentioned colors become a hard filter) ---
        # Covers all colors in the dataset: black, white, brown, golden, gray, grey,
        # orange, cream, tan, red, green, blue, silver, sable, fawn, beige
        # Also handles "black and red" → ['black', 'red']
        color_words = {
            'black': 'black', 'white': 'white', 'brown': 'brown',
            'golden': 'golden', 'gray': 'gray', 'grey': 'grey',
            'orange': 'orange', 'cream': 'cream', 'tan': 'tan',
            'red': 'red', 'green': 'green', 'blue': 'blue',
            'silver': 'silver', 'sable': 'sable', 'fawn': 'fawn',
            'beige': 'beige',
        }
        found_colors = []
        for cw in color_words:
            if re.search(r'\b' + re.escape(cw) + r'\b', q):
                found_colors.append(cw)
        if found_colors:
            prefs['color'] = found_colors  # list of color strings
        
        # --- Previous owner ---
        if re.search(r'\bfirst[\s-]?time\b', q) or re.search(r'\bno previous owner\b', q) or re.search(r'\bnever (been )?owned\b', q):
            prefs['has_previous_owner'] = False
        elif re.search(r'\bpreviously owned\b', q) or re.search(r'\bpre[\s-]?owned\b', q):
            prefs['has_previous_owner'] = True
        
        return prefs
    
    def _extract_keywords(self, query):
        """Extract physical attribute keywords from query with synonym-aware
        word-boundary matching.
        
        Key design choices:
        - ALL matching uses \\b word boundaries (no substring false-positives)
        - Synonym expansion: 'furry'→'fur', 'quadruped'→'four legs', etc.
        - Singular→plural normalization: 'leg'→'legs', 'feather'→'feathers', etc.
        - Color keywords extracted (pet data has color field)
        - Size synonyms normalized: big→'large', tiny→'small'
        - Conflicting sizes: LAST-mentioned wins (matching _extract_attribute_preferences)
        """
        import re
        
        query_lower = query.lower()
        
        # Normalize numbers to words
        query_lower = query_lower.replace('2', 'two').replace('4', 'four')
        
        # Singular → plural normalization (word boundary safe)
        SINGULAR_PLURAL = [
            (r'\bleg\b', 'legs'), (r'\bfoot\b', 'feet'), (r'\bpaw\b', 'paws'),
            (r'\bwing\b', 'wings'), (r'\bfeather\b', 'feathers'), (r'\bclaw\b', 'claws'),
            (r'\bhoof\b', 'hooves'), (r'\bear\b', 'ears'), (r'\bwhisker\b', 'whiskers'),
            (r'\bscale\b', 'scales'),
        ]
        for pattern, replacement in SINGULAR_PLURAL:
            query_lower = re.sub(pattern, replacement, query_lower)
        
        # Fix common misspellings / double plurals
        query_lower = query_lower.replace('feets', 'feet').replace('foots', 'feet')
        query_lower = query_lower.replace('legss', 'legs').replace('pawss', 'paws')
        
        found_keywords = []
        found_set = set()  # for dedup
        
        # --- 1. Synonym expansion: user term → standard keyword ---
        # e.g. "furry" → "fur", "quadruped" → "four legs"
        SYNONYM_MAP = {
            'bipedal': 'two legs', 'biped': 'two legs',
            'quadruped': 'four legs',
            'feathered': 'feathers', 'feathery': 'feathers', 'plumage': 'feathers',
            'winged': 'wings', 'flying': 'wings', 'flies': 'wings', 'fly': 'wings',
            'furry': 'fur', 'fluffy': 'fur', 'hairy': 'fur',
            'scaly': 'scales', 'scaled': 'scales',
            'talons': 'claws',
            'swimming': 'fins', 'swims': 'fins',  # no pet has fins → correctly returns empty
        }
        for syn_word, standard_kw in SYNONYM_MAP.items():
            if re.search(r'\b' + re.escape(syn_word) + r'\b', query_lower):
                if standard_kw not in found_set:
                    found_keywords.append(standard_kw)
                    found_set.add(standard_kw)
        
        # --- 2. Standard physical keywords (word-boundary matching) ---
        physical_keywords = [
            'two feet', 'two legs',
            'four legs', 'four paws',
            'wings', 'feathers', 'beak',
            'tail', 'whiskers', 'fur', 'scales', 'claws', 'hooves', 'ears'
        ]
        for keyword in physical_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower) and keyword not in found_set:
                found_keywords.append(keyword)
                found_set.add(keyword)
        
        # --- 3. Color keywords (common pet colors) ---
        color_keywords = [
            'black', 'white', 'brown', 'golden', 'gray', 'grey', 'orange',
            'cream', 'tan', 'red', 'blue', 'silver', 'sable', 'spotted',
            'tabby', 'calico', 'brindle', 'merle', 'tricolor',
        ]
        for color in color_keywords:
            if re.search(r'\b' + re.escape(color) + r'\b', query_lower):
                if color not in found_set:
                    found_keywords.append(color)
                    found_set.add(color)
        
        # --- 4. Size: resolve synonyms and pick LAST-mentioned ---
        # "big"→"large", "tiny"/"compact"→"small" so keywords match pet database values
        size_patterns = [
            (r'\bsmall\b', 'small'), (r'\btiny\b', 'small'), (r'\bcompact\b', 'small'),
            (r'\bmedium\b', 'medium'),
            (r'\blarge\b', 'large'), (r'\bbig\b', 'large'),
        ]
        last_size = None
        last_size_pos = -1
        for pattern, size_val in size_patterns:
            for m in re.finditer(pattern, query_lower):
                if m.start() > last_size_pos:
                    last_size_pos = m.start()
                    last_size = size_val
        
        # Only add the ONE resolved size keyword (no contradictions)
        if last_size and last_size not in found_set:
            found_keywords.append(last_size)
            found_set.add(last_size)
        
        return found_keywords
    
    def _keyword_match_score(self, text, keywords):
        """Calculate keyword match score (0.0 to 1.0) using word-boundary
        matching and cross-synonym awareness.
        
        Key: pet_characteristics data uses specific vocabulary:
          - Dogs: "thick double coat" (NOT "fur")
          - Cats: "long silky fur" or "short coat"
          - Birds: "two feet, wings, feathers"
          - Rabbits: "soft fur, long ears"
        
        So keyword 'fur' must also check for 'coat' in the text, and vice versa.
        """
        import re
        
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        # Cross-synonyms: if keyword is X, also check for Y in text.
        # These cover vocabulary mismatches between user queries and pet data.
        CROSS_SYNONYMS = {
            'two legs':  ['two feet', 'bipedal'],
            'two feet':  ['two legs', 'bipedal'],
            'four legs': ['four paws', 'quadruped'],
            'four paws': ['four legs', 'quadruped'],
            'fur':       ['coat', 'fluffy', 'furry'],      # dogs use "coat", cats/rabbits use "fur"
            'feathers':  ['feathered', 'plumage'],
            'wings':     ['winged'],
            'scales':    ['scaly'],
            'claws':     ['talons'],
            'gray':      ['grey'],
            'grey':      ['gray'],
        }
        
        for kw in keywords:
            # Primary check: word-boundary match
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches += 1
            else:
                # Fallback: check cross-synonyms
                found_syn = False
                for syn in CROSS_SYNONYMS.get(kw, []):
                    syn_pattern = r'\b' + re.escape(syn) + r'\b'
                    if re.search(syn_pattern, text_lower):
                        matches += 1
                        found_syn = True
                        break
        
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
        
        # Add age_years as a convenience field, but KEEP age_months as total months
        if 'age_months' in pet_copy and 'age_years' not in pet_copy:
            total_months = pet_copy['age_months']
            pet_copy['age_years'] = total_months // 12
            pet_copy['age_remaining_months'] = total_months % 12
            # age_months stays as total months (NOT overwritten)

        # Compute days_in_shelter dynamically from shelter_entry_date
        if 'shelter_entry_date' in pet_copy:
            entry = date.fromisoformat(pet_copy['shelter_entry_date'])
            pet_copy['days_in_shelter'] = (date.today() - entry).days

        return pet_copy
    
    def _quiz_to_features(self, answers):
        """
        Convert quiz answers to feature vector matching training data
        
        Expected quiz answers:
        - pet_type: 0=bird, 1=cat, 2=dog, 3=rabbit (optional hard filter)
        - size_preference: 0=large, 1=medium, 2=small
        - energy_level: 0=high, 1=low, 2=moderate
        - has_kids: true/false
        - vaccinated_important: true/false
        - shedding_tolerance: 0-5 (higher = more tolerant)
        - meat_diet_preference: 'yes'/'no'/'any'
        - age_preference: 0=young (<12 months), 1=adult (12-60), 2=senior (60+)
        - home_type: 'Apartment'|'House with small yard'|'House with large yard'|'Farm/Rural property'
        - health_preference: 0=open to health issues, 1=no health issues, 2=depends
        
        Feature order must match training (9 features): 
        ['Size', 'EnergyLevel', 'kid_friendliness', 'Vaccinated',
         'shedding', 'MeatConsumption', 'AgeMonths', 'WeightKg',
         'HealthCondition']
        """
        
        # Map answers to features
        size = answers.get('size_preference', 1)  # default medium
        energy_level = answers.get('energy_level', 2)  # default moderate
        kid_friendly = 1 if answers.get('has_kids', False) else 0
        vaccinated = 1 if answers.get('vaccinated_important', True) else 0
        shedding = answers.get('shedding_tolerance', 3)  # 0-5 range
        meat_pref = answers.get('meat_diet_preference', 'any')
        meat_consumption = 1 if meat_pref == 'yes' else (0 if meat_pref == 'no' else 1)
        
        # Home type is informational context only — the user's explicit
        # size and energy answers are respected as-is.  Previously this
        # block silently replaced medium→small (apartment) or medium→large
        # (large yard), overriding the user's actual choice.
        
        # Age preference to normalized age months
        # Training CSV AgeMonths is Z-score normalized with mean=81.56mo, std=57.85mo
        # Correct Z-scores:  6mo→-1.31, 36mo→-0.79, 108mo→+0.46
        age_pref = answers.get('age_preference', 1)
        if age_pref == 0:  # young  (target ≈6 months)
            age_months = -1.3
        elif age_pref == 2:  # senior (target ≈108 months / 9 years)
            age_months = 0.5
        else:  # adult (target ≈36 months / 3 years)
            age_months = -0.8
        
        # Weight - estimate based on size preference
        # Training CSV WeightKg is Z-score normalized with mean=6.86kg, std=8.79kg
        # Correct Z-scores: 30kg→+2.63, 8kg→+0.13, 3kg→-0.44
        weight_map = {0: 2.5, 1: 0.1, 2: -0.5}  # large, medium, small (normalized)
        weight = weight_map.get(size, 0.1)
        
        # Health condition preference
        # Dataset: 0 = Excellent, 1 = Good (has conditions)
        health_pref = answers.get('health_preference', 2)  # default: depends
        if health_pref == 1:  # user wants no health issues → prefer Excellent (0)
            health_condition = 0
        elif health_pref == 0:  # user is open to health issues → prefer Good (1)
            health_condition = 1
        else:  # depends → neutral middle
            health_condition = 0  # default to excellent
        
        return [size, energy_level, kid_friendly, vaccinated, 
                shedding, meat_consumption, age_months, weight,
                health_condition]
    
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
        
        # Check health condition
        health_pref = answers.get('health_preference')
        if health_pref == 1 and pet.get('health_condition') == 'Excellent':
            reasons.append("excellent health")
        elif health_pref == 0:
            reasons.append("open to all health conditions")
        
        if not reasons:
            reasons.append("good overall match based on your preferences")
        
        return "Perfect match: " + ", ".join(reasons[:3]) + "!"
    
    def _build_pet_text(self, pet):
        """
        Build SBERT embedding text for a pet using the SAME format as training
        (enhanced_training.py) so custom pets live in the same embedding space
        as original dataset pets.
        """
        text = f"{pet['type']} {pet.get('breed', '')}. "
        text += f"{pet.get('gender', '')} {pet.get('size', 'medium').lower()} size, {pet.get('color', '').lower()} color. "
        text += f"{pet.get('age_months', 0)} months old, weighs {pet.get('weight_kg', 0)}kg. "
        text += f"Energy level: {pet.get('energy_level', 'Moderate')}. "
        
        # Diet info
        if pet.get('meat_consumption'):
            text += "Meat-based diet. "
        else:
            text += "Non-meat diet, herbivore. "
        
        if pet.get('food_preference'):
            text += f"Diet: {pet['food_preference']}. "
        
        # Kid friendliness
        if pet.get('kid_friendly'):
            text += "Great with kids, family-friendly. "
        else:
            text += "Not ideal for young kids. "
        
        # Vaccination status
        if pet.get('vaccinated'):
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
        
        # Shedding info
        shed = pet.get('shedding_level', 0)
        shedding_desc = {0: "No shedding", 1: "Very low shedding", 2: "Low shedding",
                         3: "Moderate shedding", 4: "High shedding", 5: "Heavy shedding"}
        text += f"{shedding_desc.get(shed, 'Unknown shedding')}. "
        
        # Personality description
        if pet.get('description'):
            text += f"Personality: {pet['description']}. "
        
        # Physical characteristics
        if pet.get('pet_characteristics'):
            text += f"Physical characteristics: {pet['pet_characteristics']}. "
        
        return text
    
    def register_custom_pet(self, pet_dict):
        """
        Register a newly-added custom pet into the live engine so it
        appears in browse, quiz, and text-search results immediately
        (no retrain needed).
        
        pet_dict must follow the same schema as pets_database entries
        (keys: id, name, type, breed, age_months, size, ..., raw_features).
        """
        # --- 1. Add to pets_database ---
        self.pets_database.append(pet_dict)
        
        # --- 2. SBERT embedding (uses SAME text format as training) ---
        if self.sbert_model is not None and self.sbert_embeddings is not None:
            text = self._build_pet_text(pet_dict)
            new_emb = self.sbert_model.encode([text])  # shape (1, dim)
            self.sbert_embeddings = np.vstack([self.sbert_embeddings, new_emb])

            # Also add description embedding
            desc = pet_dict.get('description', '') or ''
            desc_emb = self.sbert_model.encode([desc])
            if self.sbert_desc_embeddings is not None:
                self.sbert_desc_embeddings = np.vstack([self.sbert_desc_embeddings, desc_emb])

        # --- 3. KNN features (no scaler — data is already normalized) ---
        if self.knn_X is not None:
            raw = pet_dict.get('raw_features', {})
            if raw and self.feature_columns:
                feat_vec = np.array([[raw.get(col, 0) for col in self.feature_columns]], dtype=float)
                self.knn_X = np.vstack([self.knn_X, feat_vec])
        
        print(f"✅ Registered custom pet #{pet_dict['id']} into recommendation engine "
              f"(total: {len(self.pets_database)})")
    
    def update_custom_pet(self, pet_id, pet_dict):
        """
        Update an existing custom pet in the live engine.
        Replaces its pets_database entry, SBERT embedding, and KNN features.
        """
        # --- 1. Find the pet's index in pets_database ---
        idx = None
        for i, p in enumerate(self.pets_database):
            if p.get('id') == pet_id:
                idx = i
                break
        
        if idx is None:
            print(f"⚠️ Pet #{pet_id} not found in engine for update")
            return False
        
        # --- 2. Replace in pets_database ---
        pet_dict['index'] = idx
        self.pets_database[idx] = pet_dict
        
        # --- 3. Recompute SBERT embedding (uses SAME text format as training) ---
        if self.sbert_model is not None and self.sbert_embeddings is not None:
            text = self._build_pet_text(pet_dict)
            new_emb = self.sbert_model.encode([text])  # shape (1, dim)
            self.sbert_embeddings[idx] = new_emb[0]

            # Also update description embedding
            if self.sbert_desc_embeddings is not None:
                desc = pet_dict.get('description', '') or ''
                desc_emb = self.sbert_model.encode([desc])
                self.sbert_desc_embeddings[idx] = desc_emb[0]

        # --- 4. Recompute KNN features (no scaler — data is already normalized) ---
        if self.knn_X is not None:
            raw = pet_dict.get('raw_features', {})
            if raw and self.feature_columns:
                feat_vec = np.array([[raw.get(col, 0) for col in self.feature_columns]], dtype=float)
                self.knn_X[idx] = feat_vec[0]
        
        print(f"✅ Updated custom pet #{pet_id} in recommendation engine")
        return True
    
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
