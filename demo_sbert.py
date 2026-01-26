#!/usr/bin/env python3
"""
Quick Demo Script for SBERT Model with New Dataset
Shows live search examples with the updated model
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_system():
    """Load SBERT system"""
    with open("model/pets_database.pkl", "rb") as f:
        pets_database = pickle.load(f)
    
    embeddings = np.load("model/sbert_embeddings.npy")
    model = SentenceTransformer("model/sbert_model")
    
    return model, embeddings, pets_database

def search(query, model, embeddings, pets_database, k=5):
    """Perform semantic search"""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    
    print(f"\n{'='*80}")
    print(f"🔍 Query: \"{query}\"")
    print(f"{'='*80}\n")
    
    for rank, idx in enumerate(top_indices, 1):
        pet = pets_database[idx]
        score = similarities[idx]
        
        print(f"{rank}. {pet['name']} - {pet['type']} ({pet['breed']})")
        print(f"   📊 Similarity: {score:.4f}")
        print(f"   📝 {pet['color']}, {pet['size']}, {pet['gender']}")
        print(f"   💭 {pet['pet_details'][:70]}...")
        print(f"   🐾 {pet['pet_characteristics'][:70]}...")
        print()

def main():
    print("\n" + "="*80)
    print("🎯 SBERT MODEL DEMO - NEW DATASET")
    print("="*80)
    print("Dataset: sbert_refined_data_with_breed_characteristics_gender_full.csv")
    print("Total Pets: 1,985 | Embedding Dimension: 384")
    print("="*80)
    
    print("\n🔄 Loading system...")
    model, embeddings, pets_database = load_system()
    print("✅ System loaded!\n")
    
    # Demo queries
    queries = [
        "I want a loyal protective German Shepherd for my family",
        "Looking for a small cuddly pet that's good with children",
        "Need an intelligent dog that's easy to train",
        "Want a fluffy cat with beautiful fur",
        "Looking for an energetic playful companion"
    ]
    
    print("="*80)
    print("🎬 LIVE DEMO - 5 Search Examples")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        input(f"\nPress Enter to run demo {i}/{len(queries)}...")
        search(query, model, embeddings, pets_database, k=3)
    
    print("\n" + "="*80)
    print("✨ DEMO COMPLETE")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Semantic understanding of natural language queries")
    print("  ✓ Rich personality and characteristic matching")
    print("  ✓ Gender-aware recommendations")
    print("  ✓ 1,985 pets with detailed descriptions")
    print("  ✓ Fast cosine similarity search")
    
    print("\n💡 Try your own queries:")
    print("  python -c \"from test_sbert_search import *; main()\"")
    print("\nOr run: uv run test_sbert_search.py")

if __name__ == "__main__":
    main()
