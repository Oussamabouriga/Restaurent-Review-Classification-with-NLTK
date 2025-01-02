# Restaurant Review Classification with NLTK

## Overview
This repository contains a Jupyter Notebook that demonstrates how to classify restaurant reviews as either **positive** (liked) or **negative** (not liked) using the Natural Language Toolkit (NLTK). The notebook walks through data preprocessing, text cleaning, and applying machine learning techniques for sentiment analysis.

---

## Features
- **Data Loading**:
  - Loads a dataset of restaurant reviews from a TSV file.
- **Data Preprocessing**:
  - Removes unnecessary characters and punctuation.
  - Tokenizes and stems words using NLTK.
  - Converts text into a structured format for machine learning.
- **Exploratory Data Analysis (EDA)**:
  - Visualizes patterns and statistics in the review data.
- **Model Training and Evaluation**:
  - Trains a classification model to predict review sentiment.
  - Evaluates model performance using accuracy and other metrics.
- **Prediction**:
  - Predicts the sentiment of new restaurant reviews.

---

## Getting Started

### Prerequisites
To run this notebook, you need:
- Python 3.8+
- Jupyter Notebook
- Required Python libraries (see below).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/restaurant-review-classification.git
   cd restaurant-review-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
The dataset `Restaurant_Reviews.tsv` contains two columns:
- `Review`: Text of the review.
- `Liked`: Binary label (1 for positive review, 0 for negative review).

Place the dataset in the same directory as the notebook or update the file path in the code.

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Restaurent_Review_Classification_with_NLTK.ipynb
   ```
2. Run the cells sequentially to:
   - Load and clean the data.
   - Process the reviews using NLTK.
   - Train a machine learning model.
   - Test the model on new reviews.

---

## Libraries Used
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **NLTK**: For natural language processing tasks.
- **Scikit-learn**: For machine learning models.

Install these libraries using:
```bash
pip install pandas numpy matplotlib nltk scikit-learn
```

---

## Results
- Achieved good classification performance with high accuracy.
- Demonstrates practical text processing for sentiment analysis.



