# 💰 ProjectAlpha — Income Prediction ML Pipeline

An end-to-end machine learning pipeline that predicts whether an individual's annual income exceeds **$50K** based on U.S. Census demographic data. The project provides a fully modular ML pipeline with a FastAPI-powered REST API for real-time predictions.

---

## 🎯 Goal

The goal of this project is to build a production-ready ML pipeline that:
- Ingests and preprocesses the **Adult Census Income** dataset
- Trains a classification model to predict income level (`>50K` or `<=50K`)
- Exposes predictions through a **REST API** for real-time inference

---

## 📊 Dataset

The project uses the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) (also known as the "adult" dataset).

| Column | Description | Type |
|--------|-------------|------|
| `age` | Age of the individual | Numerical |
| `workclass` | Type of employment (Private, Self-emp, etc.) | Categorical |
| `fnlwgt` | Final census sampling weight | Numerical |
| `education` | Highest education level | Categorical |
| `education.num` | Numeric representation of education | Numerical |
| `marital.status` | Marital status | Categorical |
| `occupation` | Type of job | Categorical |
| `relationship` | Relationship status | Categorical |
| `race` | Race | Categorical |
| `sex` | Gender | Categorical |
| `capital.gain` | Income from investments | Numerical |
| `capital.loss` | Loss from investments | Numerical |
| `hours.per.week` | Hours worked per week | Numerical |
| `native.country` | Country of origin | Categorical |
| `income` | **Target** — `>50K` or `<=50K` | Categorical |

---

## 🏗️ Project Architecture

```
ProjectAlpha/
├── Notebooks/
│   └── Data/
│       └── adult.csv               # Raw dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Loads & splits data into train/test CSVs
│   │   ├── data_transformation.py  # Preprocessing (scaling, encoding, imputation)
│   │   └── model_trainer.py        # Trains & evaluates the model
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Orchestrates full training flow
│   │   ├── predict_pipeline.py     # Loads model & runs inference
│   │   └── utils.py                # Helper functions (save/load pickle, CSV)
│   ├── logger.py                   # Centralized logging
│   └── exception.py                # Custom exception handler
├── artifacts/                      # Generated: model.pkl, preprocessor.pkl, CSVs
├── Templates/
│   └── index.html                  # (Optional) Frontend HTML template
├── application.py                  # FastAPI app (REST API)
├── config.yaml                     # Paths & hyperparameter configuration
├── requirements.txt                # Python dependencies
└── setup.py                        # Package setup
```

### Pipeline Flow

```
adult.csv
    │
    ▼
data_ingestion      →  artifacts/train.csv, test.csv
    │
    ▼
data_transformation →  artifacts/preprocessor.pkl  (ColumnTransformer)
    │
    ▼
model_trainer       →  artifacts/model.pkl  (LogisticRegression, AUC ~0.90)
    │
    ▼
FastAPI /predict    →  {"prediction": ">50K"} or {"prediction": "<=50K"}
```

---

## ⚙️ Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~84.8% |
| ROC-AUC | ~90.3% |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ProjectAlpha
```

### 2. Create and activate a virtual environment

```bash
python -m venv income
source income/Scripts/activate   # Git Bash on Windows
# OR
income\Scripts\activate           # PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training pipeline

```bash
python -m src.pipeline.train_pipeline
```

This will:
- Read `Notebooks/Data/adult.csv`
- Save `artifacts/train.csv` and `artifacts/test.csv`
- Save `artifacts/preprocessor.pkl`
- Save `artifacts/model.pkl`
- Print accuracy and ROC-AUC scores

### 5. Start the API server

```bash
uvicorn application:app --reload
```

The API will be available at **http://127.0.0.1:8000**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict income class |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | API documentation |

### Example `/predict` Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 37,
    "workclass": "Private",
    "fnlwgt": 284582,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
  }'
```

**Response:**
```json
{"prediction": ">50K"}
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML | scikit-learn |
| Data | pandas, numpy |
| API | FastAPI + Uvicorn |
| Serialization | joblib |
| Logging | Python logging |

---

## 📝 Running Individual Components

```bash
python -m src.components.data_ingestion       # Ingest data only
python -m src.components.model_trainer        # Train model only
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.