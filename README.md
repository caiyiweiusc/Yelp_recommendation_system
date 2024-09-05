# Hybrid Recommendation System

## Project Overview
This project implements a high-performance hybrid recommendation system using Spark and XGBoost, designed for rapid processing of large-scale datasets. It combines item-based collaborative filtering with a model-based approach to predict user ratings for businesses. The system efficiently utilizes the Yelp dataset, incorporating user, business, review, check-in, and photo data to generate accurate predictions quickly.

## Features
- Rapid, large-scale data processing capability using PySpark
- Efficient implementation of item-based collaborative filtering algorithm
- Fast machine learning model training using XGBoost
- Seamless integration of multiple data sources from Yelp: user information, business details, reviews, check-ins, and photos
- Hybrid recommendation strategy balancing collaborative filtering and machine learning model results for optimal performance
- Designed for scalability, capable of handling datasets with millions of records

## Installation
1. Ensure Python 3.x and Java are installed on your system
2. Install required Python libraries:
   ```
   pip install pyspark xgboost numpy
   ```
3. Download and configure the Spark environment

## Usage
1. Prepare your data files and ensure they are in the correct directory
2. Run the script with the necessary parameters:
   ```
   spark-submit hybrid_recommendation_system.py <folder_path> <test_file> <output_file>
   ```
   Where:
   - `<folder_path>`: Path to the folder containing training data
   - `<test_file>`: Path to the test data file
   - `<output_file>`: Path for the output file with prediction results

## Code Structure
- `parse_*` functions: Efficiently parse different types of data (photos, check-ins, businesses, users, etc.)
- `calculate_similarity_score`: Rapidly calculate similarity between businesses
- `estimate_rating_item_based`: Quickly estimate ratings based on item similarity
- `build_features`: Construct feature vectors optimized for performance
- `main` function: Orchestrates the entire process, ensuring efficient data flow and processing

## Performance
The system leverages Spark for parallel computing, enabling it to handle large-scale datasets with exceptional speed. In benchmark tests:

- Successfully processed and generated predictions for 20,000 Yelp dataset entries
- Achieved a Root Mean Square Error (RMSE) of 0.97, demonstrating high accuracy
- Execution time scales efficiently with data size, making it suitable for datasets with millions of records

The combination of Spark's distributed computing capabilities and XGBoost's efficient algorithm implementation allows for rapid training and prediction, even on large datasets.

## Detailed Algorithm Description

### Item-Based Collaborative Filtering
The system implements a highly optimized item-based collaborative filtering approach, which involves:
1. Efficiently calculating similarity scores between businesses based on user ratings
2. Utilizing these similarity scores to rapidly estimate ratings for new user-business pairs

### XGBoost Model
A gradient boosting model is trained using XGBoost with the following features, optimized for quick processing:
- Business features: stars, review count, average usefulness/funniness/coolness of reviews, total check-ins, total photos
- User features: review count, average stars, squared fan count, elite status, sum of squared compliments

### Hybrid Approach
The final prediction is a weighted combination of the item-based and model-based predictions, allowing for a balance between different recommendation strategies while maintaining high processing speed.

## Future Improvements
- Implement advanced feature engineering techniques for better prediction accurancy
- Investigate techniques for handling even larger datasets (100M+ records) efficiently
