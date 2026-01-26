#!/usr/bin/env python3
"""
Final Integration Test for SBERT Model with New Dataset
Comprehensive end-to-end test to verify everything works correctly
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def test_model_files():
    """Test that all required model files exist and are valid"""
    print("="*80)
    print("🔍 TEST 1: Checking Model Files")
    print("="*80)
    
    model_dir = "model"
    required_files = [
        "pets_database.pkl",
        "sbert_embeddings.npy",
        "sbert_model"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✅ All model files exist!")
        return True
    else:
        print("\n❌ Some model files are missing!")
        return False

def test_database_loading():
    """Test loading the pets database"""
    print("\n" + "="*80)
    print("🔍 TEST 2: Loading Pets Database")
    print("="*80)
    
    try:
        with open("model/pets_database.pkl", "rb") as f:
            pets_database = pickle.load(f)
        
        print(f"  ✓ Successfully loaded database")
        print(f"  ✓ Total pets: {len(pets_database)}")
        
        # Verify structure
        if len(pets_database) > 0:
            sample_pet = pets_database[0]
            required_keys = ['pet_id', 'name', 'type', 'breed', 'color', 'size', 'gender', 'pet_details', 'pet_characteristics']
            
            has_all_keys = all(key in sample_pet for key in required_keys)
            if has_all_keys:
                print(f"  ✓ Database structure is correct")
                print(f"\n  Sample pet:")
                print(f"    Name: {sample_pet['name']}")
                print(f"    Type: {sample_pet['type']}")
                print(f"    Breed: {sample_pet['breed']}")
                print(f"    Details: {sample_pet['pet_details'][:60]}...")
            else:
                print(f"  ✗ Database structure is incorrect")
                return False
        
        print("\n✅ Database loading test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading database: {e}")
        return False

def test_embeddings_loading():
    """Test loading SBERT embeddings"""
    print("\n" + "="*80)
    print("🔍 TEST 3: Loading SBERT Embeddings")
    print("="*80)
    
    try:
        embeddings = np.load("model/sbert_embeddings.npy")
        print(f"  ✓ Successfully loaded embeddings")
        print(f"  ✓ Shape: {embeddings.shape}")
        print(f"  ✓ Number of pets: {embeddings.shape[0]}")
        print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
        
        if embeddings.shape[1] == 384:
            print(f"  ✓ Correct embedding dimension (384)")
        else:
            print(f"  ✗ Incorrect embedding dimension (expected 384, got {embeddings.shape[1]})")
            return False
        
        print("\n✅ Embeddings loading test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading embeddings: {e}")
        return False

def test_model_loading():
    """Test loading SBERT model"""
    print("\n" + "="*80)
    print("🔍 TEST 4: Loading SBERT Model")
    print("="*80)
    
    try:
        model = SentenceTransformer("model/sbert_model")
        print(f"  ✓ Successfully loaded SBERT model")
        
        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"  ✓ Model can encode text")
        print(f"  ✓ Test embedding shape: {embedding.shape}")
        
        print("\n✅ Model loading test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False

def test_search_functionality():
    """Test the complete search functionality"""
    print("\n" + "="*80)
    print("🔍 TEST 5: Search Functionality")
    print("="*80)
    
    try:
        # Load everything
        with open("model/pets_database.pkl", "rb") as f:
            pets_database = pickle.load(f)
        
        embeddings = np.load("model/sbert_embeddings.npy")
        model = SentenceTransformer("model/sbert_model")
        
        # Test search
        test_queries = [
            "loyal intelligent dog",
            "cuddly friendly pet",
            "small fluffy cat"
        ]
        
        print("\n  Testing search queries:\n")
        
        for query in test_queries:
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            top_idx = np.argmax(similarities)
            
            pet = pets_database[top_idx]
            score = similarities[top_idx]
            
            print(f"  Query: '{query}'")
            print(f"    → Top result: {pet['name']} ({pet['type']}, {pet['breed']})")
            print(f"    → Similarity: {score:.4f}")
            print()
        
        print("✅ Search functionality test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error in search test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test data consistency between database and embeddings"""
    print("\n" + "="*80)
    print("🔍 TEST 6: Data Consistency")
    print("="*80)
    
    try:
        with open("model/pets_database.pkl", "rb") as f:
            pets_database = pickle.load(f)
        
        embeddings = np.load("model/sbert_embeddings.npy")
        
        db_size = len(pets_database)
        emb_size = embeddings.shape[0]
        
        print(f"  Database size: {db_size}")
        print(f"  Embeddings size: {emb_size}")
        
        if db_size == emb_size:
            print(f"  ✓ Sizes match perfectly!")
        else:
            print(f"  ✗ Size mismatch!")
            return False
        
        # Check for expected dataset size
        if db_size == 1985:
            print(f"  ✓ Correct dataset size (1,985 pets)")
        else:
            print(f"  ⚠ Unexpected dataset size (expected 1,985, got {db_size})")
        
        print("\n✅ Data consistency test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error in consistency test: {e}")
        return False

def test_new_dataset_features():
    """Test that new dataset features are present"""
    print("\n" + "="*80)
    print("🔍 TEST 7: New Dataset Features")
    print("="*80)
    
    try:
        with open("model/pets_database.pkl", "rb") as f:
            pets_database = pickle.load(f)
        
        sample_pet = pets_database[0]
        
        # Check for gender field
        if 'gender' in sample_pet:
            print(f"  ✓ Gender field present: {sample_pet['gender']}")
        else:
            print(f"  ✗ Gender field missing")
            return False
        
        # Check for detailed pet_details
        if 'pet_details' in sample_pet and len(sample_pet['pet_details']) > 20:
            print(f"  ✓ Detailed pet_details present ({len(sample_pet['pet_details'])} chars)")
        else:
            print(f"  ✗ pet_details missing or too short")
            return False
        
        # Check for pet_characteristics
        if 'pet_characteristics' in sample_pet and len(sample_pet['pet_characteristics']) > 20:
            print(f"  ✓ pet_characteristics present ({len(sample_pet['pet_characteristics'])} chars)")
        else:
            print(f"  ✗ pet_characteristics missing or too short")
            return False
        
        print("\n✅ New dataset features test passed!")
        return True
    
    except Exception as e:
        print(f"  ✗ Error in features test: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 SBERT MODEL - COMPREHENSIVE INTEGRATION TEST")
    print("="*80)
    print("Testing updated SBERT model with new dataset")
    print("Dataset: sbert_refined_data_with_breed_characteristics_gender_full.csv")
    print("="*80)
    
    tests = [
        test_model_files,
        test_database_loading,
        test_embeddings_loading,
        test_model_loading,
        test_search_functionality,
        test_data_consistency,
        test_new_dataset_features
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n" + "🎉"*20)
        print("✅ ALL TESTS PASSED!")
        print("🎉"*20)
        print("\nThe SBERT model is working perfectly with the new dataset!")
        print("You can now use it for text-based pet recommendations.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the errors above.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
