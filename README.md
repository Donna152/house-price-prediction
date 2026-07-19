# 🏠 House Price Predictor

A machine learning web app that predicts residential property prices in Bengaluru based on location, size (BHK), total square footage, and number of bathrooms. Built with scikit-learn and deployed as an interactive Streamlit app.

**🔗 Live Demo:** [donna7-house-price-predictor.streamlit.app](https://donna7-house-price-predictor.streamlit.app)

---

## 📌 Overview

This project builds a regression pipeline that estimates residential property market value (in lakhs, INR) from real-world Bengaluru real estate listing data. The raw dataset is heavily cleaned — missing values imputed, inconsistent formats normalized, and unrealistic listings filtered out through multiple stages of domain-specific outlier removal — before training a regularized linear model. The entire pipeline (encoding, scaling, and model) is bundled into a single deployable object.

## ✨ Features

- 🏘️ Select the location, BHK size, total square footage, and number of bathrooms
- 💰 Get an instant estimated property price (in lakhs, INR)
- 🧹 Model trained on a heavily cleaned, multi-stage outlier-filtered version of real Bengaluru housing data
- 🔗 End-to-end scikit-learn `Pipeline` (encoding + scaling + model) for consistent predictions

## 🖼️ Screenshots

Screenshots of the app in action are available in the [`Results`](./Results) folder.

## 🧠 How It Works

### 1. Dataset

The project uses **`Bengaluru_House_Data.csv`** (in the [`dataset`](./dataset) folder) — a collection of residential real estate listings across Bengaluru, containing **13,320 entries**:

| Column | Description |
|---|---|
| `price` | **(Target)** Market value of the property in lakhs (INR) |
| `location` | Area or locality of the property within Bengaluru |
| `size` | Number of bedrooms/BHK (e.g. `"2 BHK"`) |
| `total_sqft` | Total surface area of the property in square feet |
| `bath` | Number of bathrooms |
| `balcony`, `area_type`, `availability`, `society` | Additional metadata columns (later dropped — see below) |

### 2. Data Cleaning

- **Dropped low-value columns:**
  - `society` — ~41% missing, too unreliable to impute
  - `balcony` — statistically insignificant and largely redundant with property size
  - `area_type` & `availability` — minimal correlation with price, more descriptive metadata than price drivers
- **`location`** — the single missing value was imputed with the most common locality (`"Sarjapur Road"`)
- **`size`** — missing values filled with the mode (`"2 BHK"`); a new `bhk` column was engineered by extracting the numeric bedroom count
- **`bath`** — missing values filled with the median
- **`total_sqft`** — contained ranges (e.g. `"1200-1400"`); these were converted to their average value via a custom parsing function

### 3. Feature Engineering & Multi-Stage Outlier Removal

Property pricing data is notoriously messy, so several outlier-removal passes were applied:

1. **Implausible BHK filter** — removed listings with unrealistic bedroom counts (e.g. a "43 BHK" property)
2. **`price_per_sqft` engineering** — a temporary `price_per_sqft` column was computed purely to power outlier detection (removed later to prevent data leakage)
3. **Rare location grouping** — locations with 10 or fewer listings were grouped into an `"other"` category to reduce dimensionality and prevent overfitting on sparse areas
4. **Minimum room size filter** — removed properties where `total_sqft / bhk < 300`, since a residential bedroom in Bengaluru is expected to be at least 300 sq ft
5. **Per-location statistical outlier removal** — for each location, listings with `price_per_sqft` outside one standard deviation of that location's average were discarded
6. **BHK price-consistency filter** — removed properties where a larger unit costs *less* per square foot than a smaller unit in the same location (a logical pricing inconsistency)

After all cleaning and filtering stages, the dataset was reduced from **13,320 → 7,416 listings**. The temporary `size` and `price_per_sqft` columns were then dropped, and the cleaned data was exported to `Cleaned_data.csv`.

### 4. Feature Encoding & Scaling

- **`location`** (categorical) is transformed using **One-Hot Encoding** (`OneHotEncoder`) via a `ColumnTransformer`
- All remaining numerical features (`total_sqft`, `bath`, `bhk`) pass through unchanged and are then **standardized** with `StandardScaler`

### 5. Model Building & Selection

Three regression algorithms were trained and compared, each wrapped in an identical `ColumnTransformer → StandardScaler → Model` **`Pipeline`**, using an 80/20 train-test split:

| Model | R² Score |
|---|---|
| Linear Regression (no regularization) | 0.8012 |
| Lasso Regression | **0.8012** |
| **Ridge Regression** | **0.8012** |

**Ridge Regression** was selected as the final model — its L2 regularization helps stabilize coefficients across the many one-hot encoded location features without sacrificing predictive accuracy.

### 6. Deployment Preparation

The final trained pipeline and cleaned dataset are serialized with `pickle`:

| File | Contents |
|---|---|
| `RidgeModel.joblib` | Full trained pipeline (One-Hot Encoding + Scaling + Ridge Regression model) |
| `data.joblib` | Cleaned housing dataframe, used to populate dropdown/input options (locations, BHK, etc.) in the app |

The Streamlit app loads both files at startup — `data.pkl` to populate the input fields, and `RidgeModel.pkl` to make predictions directly from raw input (no manual encoding needed, since it's baked into the pipeline).

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** — data cleaning and processing
- **scikit-learn** — `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `LinearRegression`, `Lasso`, `Ridge`, `Pipeline`
- **matplotlib / seaborn** — exploratory data analysis
- **Streamlit** — web app frontend
- **joblib** — model/data serialization

## 📁 Project Structure

```

├── app.py                         # Streamlit frontend
├── house_price_predictor.ipynb    # Data cleaning, EDA & model-building notebook
├── RidgeModel.joblib                 # Serialized trained pipeline (encoder + scaler + Ridge model)
├── data.joblib                       # Serialized cleaned housing dataframe (for input options)
├── dataset/                       # Raw dataset (Bengaluru_House_Data.csv)
├── Results/                       # Screenshots of app results
├── requirements.txt               # Python dependencies
├── setup.sh                       # Streamlit config setup
├── Procfile                       # Process file for deployment
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Donna152/house-price-predictor.git
   cd house-price-predictor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `RidgeModel.pkl` and `data.pkl` are present in the project root. If not, run through `house_price_predictor.ipynb` to regenerate them from `dataset/Bengaluru_House_Data.csv`.

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in your terminal (typically `http://localhost:8501`).

## 🌐 Deployment

This app is deployed on **Streamlit Community Cloud**:
👉 **[donna7-house-price-predictor.streamlit.app](https://donna7-house-price-predictor.streamlit.app)**

To deploy your own copy:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository, branch, and `app.py` as the entry point.
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` automatically.

## 📓 Notebook

`house_price_predictor.ipynb` contains the full, step-by-step pipeline used to build this system:
- Dataset loading and exploration
- Data cleaning (dropping unreliable columns, imputing missing values in `location`, `size`, and `bath`)
- Feature engineering (`bhk` extraction, `total_sqft` range parsing, temporary `price_per_sqft` for outlier detection)
- Multiple stages of domain-specific and statistical outlier removal (implausible BHK counts, rare locations, minimum room size, per-location price outliers, BHK price-consistency checks)
- Feature/target split and train-test split
- One-Hot Encoding and feature scaling via `ColumnTransformer` and `StandardScaler`
- Training and comparison of Linear, Lasso, and Ridge Regression models
- Serialization of the final `RidgeModel.pkl` and `data.pkl` used by the Streamlit app

## 🔮 Possible Improvements

- Try tree-based models (Random Forest, Gradient Boosting, XGBoost) to capture non-linear relationships between location, size, and price
- Tune the Ridge/Lasso regularization strength (`alpha`) via cross-validation (`RidgeCV`/`LassoCV`) rather than using default parameters
- Add more granular location data (e.g. proximity to landmarks, public transport) to improve location-based pricing accuracy
- Add input validation in the app to prevent unrealistic combinations (e.g. extremely low `total_sqft` for a high `bhk`)

## 📄 License

This project is intended for educational and portfolio purposes. The dataset used is the [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data), publicly available on Kaggle.
