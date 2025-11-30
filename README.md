# 🐾 Pet Adoption Recommendation System

An intelligent pet adoption system that uses **dual AI approaches** to match users with their perfect pet companions:
- **KNN (K-Nearest Neighbors)**: Quiz-based personalized recommendations
- **SBERT (Sentence-BERT)**: Natural language text search

## ✨ Features

### 🤖 Dual Recommendation System
1. **Quiz-Based Matching (KNN)**
   - Answer lifestyle questions
   - Get personalized pet matches
   - Confidence scores for each match
   - Considers: size, energy level, kid-friendliness, vaccination, shedding, age preference

2. **Text-Based Search (SBERT)**
   - Describe your ideal pet in natural language
   - AI understands semantic meaning
   - Example: "I want a fluffy, friendly cat that's good with kids"
   - Filter by pet type

### 💯 Advanced Features
- **Confidence Scores**: See match percentage for each pet (0-100%)
- **Pet Names & Breeds**: Real pet information, not just numbers
- **Detailed Profiles**: Age, size, weight, gender, color, energy level, personality
- **Adoption Requests**: Submit adoption applications directly
- **Smart Filtering**: Browse by type, size, energy level
- **Beautiful UI**: Modern, responsive design

## 📊 Dataset

- **Total Pets**: 1,985
  - Dogs: 500
  - Cats: 505
  - Birds: 487
  - Rabbits: 493

### Data Mappings
```python
PetType: bird=0, cat=1, dog=2, rabbit=3
Breed: domestic shorthair=0, german shepherd=1, labrador=2, parakeet=3, 
       persian=4, pug=5, rabbit=6, retriever=7, spitz=8
Color: beige=0, black=1, black and red=2, black and tan=3, brown=4, 
       fawn=5, gray=6, green=7, orange=8, sable=9, white=10
Size: large=0, medium=1, small=2
Gender: f=0, m=1
EnergyLevel: high=0, low=1, moderate=2
```

## 🏗️ System Architecture

### Block Diagram
```
[Pet Dataset] → [Preprocessing] → [Embedding]
                                      ↓
[User Quiz] → [Feature Vector] → [KNN Model] → [Top K Pets]
                                      ↓
[User Text] → [SBERT Embedding] → [Similarity] → [Ranked Results]
                                      ↓
                            [Pet Recommendations]
                                      ↓
                          [Adoption Request System]
```

### Technology Stack
- **Backend**: Flask (Python)
- **ML Models**: 
  - Scikit-learn KNN
  - Sentence-Transformers (SBERT)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy

## 📦 Installation

### 1. Clone Repository
```bash
cd /home/silu/Pet-Adoption
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- flask==2.3.3
- flask-cors==4.0.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- sentence-transformers==2.2.2
- joblib==1.3.2
- torch==2.0.1
- transformers==4.31.0

### 3. Train Models
```bash
python backend/enhanced_training.py
```

This will:
- Load the pet dataset (1,985 pets)
- Create pet database with names and readable information
- Train KNN model for quiz recommendations
- Generate SBERT embeddings for text search
- Save all models to `model/` directory

Expected output:
```
✓ Created database with 1985 pets
✓ KNN model trained successfully
✓ SBERT embeddings created
✅ TRAINING COMPLETED SUCCESSFULLY!
```

### 4. Run Application
```bash
python app_enhanced.py
```

The server will start on: http://localhost:5000

## 🚀 Usage

### Method 1: Quiz-Based Recommendations
1. Navigate to http://localhost:5000
2. Click "Start Quiz"
3. Answer questions about your lifestyle:
   - Pet type preference (optional)
   - Size preference (small/medium/large)
   - Your activity level
   - Kids at home?
   - Vaccination importance
   - Shedding tolerance
   - Meat diet okay?
   - Age preference
4. Submit and get top 5 matches with confidence scores!

### Method 2: Text-Based Search
1. Navigate to http://localhost:5000
2. Click "Search Now"
3. Describe your ideal pet in natural language:
   - "I want a friendly, playful dog that's good with kids"
   - "Looking for a quiet, calm cat for apartment living"
   - "Need an energetic pet that loves outdoor activities"
4. Optional: Filter by pet type
5. Get AI-powered matches!

### Method 3: Browse All Pets
1. Click "Browse All"
2. Filter by type, size, or energy level
3. View all available pets

### Adopt a Pet
1. View recommendations or browse pets
2. Click "Adopt [Pet Name]" button
3. Fill out adoption form:
   - Your name, email, phone
   - Address
   - Why you want to adopt
4. Submit request!

## 📁 Project Structure

```
Pet-Adoption/
├── app_enhanced.py              # Main Flask application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── backend/
│   ├── enhanced_training.py     # Training script (KNN + SBERT)
│   └── recommendation_engine.py # Recommendation logic
│
├── model/                       # Generated models (after training)
│   ├── pets_database.pkl        # Pet information with names
│   ├── knn_model.joblib         # KNN model
│   ├── knn_scaler.joblib        # Feature scaler
│   ├── knn_features.txt         # Feature names
│   ├── sbert_embeddings.npy     # Text embeddings
│   └── sbert_model/             # SBERT model directory
│
├── dataset/
│   └── final_refined_pet.csv    # Pet dataset (1,985 pets)
│
└── frontend/
    ├── index.html               # Landing page
    ├── quiz.html                # Quiz page (KNN)
    ├── search.html              # Text search (SBERT)
    ├── browse.html              # Browse all pets
    └── styles.css               # Beautiful CSS styles
```

## 🔧 API Endpoints

### Recommendations
- `POST /api/recommend/quiz` - Get quiz-based recommendations
- `POST /api/recommend/text` - Get text-based recommendations

### Pets
- `GET /api/pets` - Get all pets (with filters)
- `GET /api/pets/<id>` - Get specific pet details
- `GET /api/stats` - Get database statistics

### Adoption
- `POST /api/adopt` - Submit adoption request
- `GET /api/adoptions` - Get all adoption requests
- `PATCH /api/adoptions/<id>` - Update adoption status

### System
- `GET /api/health` - Health check

## 🎯 How It Works

### KNN Quiz Recommendations
1. User answers quiz questions
2. Answers converted to feature vector matching training data
3. Features scaled using StandardScaler
4. KNN finds 15 nearest neighbors
5. Distance converted to confidence score (0-100%)
6. Top 5 pets returned with match reasons

### SBERT Text Search
1. User describes ideal pet in natural language
2. Text embedded using Sentence-BERT model
3. Cosine similarity calculated with all pet embeddings
4. Pets ranked by similarity score
5. Scores converted to confidence percentage
6. Top matches returned

### Confidence Score Calculation
```python
# KNN: Distance to confidence
max_distance = distances.max()
confidence = (1 - (distance / max_distance)) * 100

# SBERT: Cosine similarity to confidence  
confidence = cosine_similarity * 100
```

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, mobile
- **Modern Gradient Headers**: Eye-catching hero sections
- **Card-Based Layout**: Clean pet presentation
- **Color-Coded Tags**: Vaccinated, Kid-friendly badges
- **Smooth Animations**: Hover effects, transitions
- **Modal Forms**: Clean adoption request interface

## 📝 Example Workflows

### Workflow 1: Quiz → Match → Adopt
```
User visits homepage
  ↓
Takes quiz (answers 8 questions)
  ↓
KNN model finds best matches
  ↓
Views top 5 pets with confidence scores
  ↓
Clicks "Adopt [Pet Name]"
  ↓
Fills adoption form
  ↓
Request submitted!
```

### Workflow 2: Search → Match → Adopt
```
User visits search page
  ↓
Describes: "friendly dog good with kids"
  ↓
SBERT finds semantic matches
  ↓
Views ranked results
  ↓
Adopts favorite pet
```

## 🔍 Diagram Improvements (From Your Images)

### Suggested Block Diagram Updates:
1. **Add "Pet Database"** component with pet names/details
2. **Add "Confidence Scoring"** module showing percentage calculation
3. **Add "Adoption System"** for request handling
4. **Features include**: Age, Size, Weight, Energy, Shedding, Kid-friendliness, etc.

### Suggested Flowchart Updates:
1. After "Recommend Pet", show "View Match Scores (%)"
2. Split "Fill Form" into:
   - "View Pet Details"
   - "Submit Adoption Request"
3. Add "Success Confirmation" before End

## 🧪 Testing

### Test KNN Recommendations
```bash
# In Python or API client
import requests

response = requests.post('http://localhost:5000/api/recommend/quiz', json={
    "answers": {
        "size_preference": 1,  # medium
        "energy_level": 2,     # moderate
        "has_kids": True,
        "vaccinated_important": True,
        "shedding_tolerance": 3,
        "okay_with_meat_diet": True,
        "age_preference": 1    # adult
    },
    "top_k": 5
})

print(response.json())
```

### Test SBERT Search
```bash
response = requests.post('http://localhost:5000/api/recommend/text', json={
    "query": "I want a friendly playful dog good with kids",
    "top_k": 5
})

print(response.json())
```

## 🎓 Machine Learning Details

### KNN Model
- **Algorithm**: K-Nearest Neighbors
- **K**: 15 neighbors
- **Weights**: Distance-weighted
- **Metric**: Euclidean distance
- **Features**: Size, Energy, Kid-friendliness, Vaccinated, Shedding, Meat diet, Age, Weight

### SBERT Model
- **Model**: all-MiniLM-L6-v2
- **Embedding Size**: 384 dimensions
- **Similarity**: Cosine similarity
- **Input**: Natural language pet descriptions
- **Output**: Semantic vector representations

## 🐛 Troubleshooting

### Models not found error
```bash
# Solution: Run training first
python backend/enhanced_training.py
```

### Port already in use
```bash
# Solution: Change port in app_enhanced.py
app.run(debug=True, port=5001)  # Use different port
```

### Import errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📊 Success Metrics

- ✅ 1,985 pets with full details
- ✅ 100% pets have names (not numbers!)
- ✅ Confidence scores for every match
- ✅ Dual recommendation approaches
- ✅ Beautiful, responsive UI
- ✅ Adoption request system
- ✅ Real-time filtering and search

## 🚀 Future Enhancements

1. **Image Upload**: Pet photos
2. **User Authentication**: Secure login system
3. **Admin Dashboard**: Manage pets and adoptions
4. **Email Notifications**: Adoption status updates
5. **Advanced Filters**: Breed-specific, age range, weight range
6. **Favorites**: Save favorite pets
7. **Comparison**: Compare multiple pets side-by-side

## 👥 Contributors

Built with ❤️ for pet lovers everywhere!

## 📄 License

This project is for educational purposes.

---

**🎉 Happy Pet Matching! 🐾**

For questions or issues, please check the code comments or API documentation.
