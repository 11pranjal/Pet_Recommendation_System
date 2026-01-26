#!/usr/bin/env python3
"""
Test script for SBERT-based pet search
Tests semantic similarity search with various queries
"""

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_sbert_system():
    """Load SBERT model, embeddings, and pets database"""
    print("🔄 Loading SBERT system...")
    
    model_dir = "model"
    
    # Load pets database
    with open(os.path.join(model_dir, "pets_database.pkl"), "rb") as f:
        pets_database = pickle.load(f)
    print(f"✓ Loaded {len(pets_database)} pets from database")
    
    # Load SBERT model
    sbert_model = SentenceTransformer(os.path.join(model_dir, "sbert_model"))
    print("✓ Loaded SBERT model")
    
    # Load embeddings
    sbert_embeddings = np.load(os.path.join(model_dir, "sbert_embeddings.npy"))
    print(f"✓ Loaded embeddings: shape {sbert_embeddings.shape}")
    
    return sbert_model, sbert_embeddings, pets_database

def search_pets(query, sbert_model, sbert_embeddings, pets_database, top_k=5):
    """Search for pets using semantic similarity"""
    print(f"\n🔍 Query: \"{query}\"")
    
    # Encode query
    query_embedding = sbert_model.encode([query], convert_to_numpy=True)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, sbert_embeddings)[0]
    
    # Get top K results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Display results
    print(f"\n📋 Top {top_k} Results:")
    print("="*80)
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        pet = pets_database[idx]
        score = similarities[idx]
        
        print(f"\n{rank}. {pet['name']} (Similarity: {score:.4f})")
        print(f"   Type: {pet['type']}")
        print(f"   Breed: {pet['breed']}")
        print(f"   Details: {pet['color']}, {pet['size']}, {pet['gender']}")
        print(f"   Personality: {pet['pet_details'][:80]}...")
        print(f"   Characteristics: {pet['pet_characteristics'][:80]}...")
        
        results.append({
            'rank': rank,
            'name': pet['name'],
            'type': pet['type'],
            'breed': pet['breed'],
            'score': float(score),
            'pet': pet
        })
    
    return results

def run_test_queries(sbert_model, sbert_embeddings, pets_database):
    """Run a series of test queries"""
    
    test_queries = [
        "I want a loyal and intelligent dog for protection",
        "Looking for a friendly and playful companion",
        "Need a cuddly pet that is good with kids",
        "Want a small fluffy pet for apartment living",
        "Looking for an active dog that loves outdoor activities",
        "Need a calm and quiet pet for elderly person",
        "Want a pet with beautiful coat and elegant appearance",
        "Looking for a protective guard dog",
        "Need a gentle and affectionate pet",
        "Want a pet that is easy to train and obedient"
    ]
    
    print("\n" + "="*80)
    print("🧪 RUNNING TEST QUERIES")
    print("="*80)
    
    all_results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*80}")
        
        results = search_pets(query, sbert_model, sbert_embeddings, pets_database, top_k=5)
        all_results[query] = results
        
        if i < len(test_queries):
            input("\nPress Enter to continue to next query...")
    
    return all_results

def analyze_results(all_results):
    """Analyze and summarize test results"""
    print("\n" + "="*80)
    print("📊 RESULTS ANALYSIS")
    print("="*80)
    
    # Collect statistics
    all_scores = []
    pet_type_distribution = {}
    breed_distribution = {}
    
    for query, results in all_results.items():
        for result in results:
            all_scores.append(result['score'])
            
            pet_type = result['type']
            pet_type_distribution[pet_type] = pet_type_distribution.get(pet_type, 0) + 1
            
            breed = result['breed']
            breed_distribution[breed] = breed_distribution.get(breed, 0) + 1
    
    # Display statistics
    print(f"\nSimilarity Scores:")
    print(f"  - Mean: {np.mean(all_scores):.4f}")
    print(f"  - Min: {np.min(all_scores):.4f}")
    print(f"  - Max: {np.max(all_scores):.4f}")
    print(f"  - Std: {np.std(all_scores):.4f}")
    
    print(f"\nPet Type Distribution in Results:")
    for pet_type, count in sorted(pet_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {pet_type}: {count} ({count/len(all_scores)*100:.1f}%)")
    
    print(f"\nTop 5 Breeds in Results:")
    top_breeds = sorted(breed_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
    for breed, count in top_breeds:
        print(f"  - {breed}: {count}")
    
    print("\n✅ Analysis complete!")

def interactive_mode(sbert_model, sbert_embeddings, pets_database):
    """Interactive search mode"""
    print("\n" + "="*80)
    print("🎮 INTERACTIVE SEARCH MODE")
    print("="*80)
    print("Enter your queries to search for pets. Type 'quit' to exit.")
    
    while True:
        query = input("\n🔍 Enter your search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        search_pets(query, sbert_model, sbert_embeddings, pets_database, top_k=5)

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("🧪 SBERT PET SEARCH - TEST SUITE")
    print("="*80)
    
    # Load system
    sbert_model, sbert_embeddings, pets_database = load_sbert_system()
    
    print("\n" + "="*80)
    print("Select Test Mode:")
    print("  1. Run predefined test queries")
    print("  2. Interactive search mode")
    print("  3. Quick test (3 queries)")
    print("="*80)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        all_results = run_test_queries(sbert_model, sbert_embeddings, pets_database)
        analyze_results(all_results)
    
    elif choice == "2":
        interactive_mode(sbert_model, sbert_embeddings, pets_database)
    
    elif choice == "3":
        # Quick test with 3 queries
        quick_queries = [
            "I want a loyal and intelligent dog for protection",
            "Looking for a cuddly pet that is good with kids",
            "Want a small fluffy pet for apartment living"
        ]
        
        for query in quick_queries:
            search_pets(query, sbert_model, sbert_embeddings, pets_database, top_k=3)
            print("\n" + "-"*80)
    
    else:
        print("Invalid choice. Running quick test...")
        quick_queries = [
            "I want a loyal and intelligent dog for protection",
            "Looking for a cuddly pet that is good with kids"
        ]
        
        for query in quick_queries:
            search_pets(query, sbert_model, sbert_embeddings, pets_database, top_k=3)
            print("\n" + "-"*80)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
