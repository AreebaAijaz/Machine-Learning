# Machine Learning — Linear Regression

A simple Linear Regression model trained to predict salary based on years of work experience.

## Dataset
- **Source:** Salary_Data.csv
- **Rows:** 30
- **Features:** YearsExperience, Salary

## What This Project Does
- Loads and explores the dataset
- Visualizes the relationship between experience and salary
- Trains a Linear Regression model on 80% of the data
- Evaluates the model on the remaining 20% (unseen data)
- Predicts salary for new custom inputs

## Model Performance
- **R² Score:** 0.89
- **RMSE:** ~$7,893

## Tools Used
- Python
- Pandas
- Matplotlib
- Scikit-learn (sklearn)

## Key Finding
There is a strong linear relationship between years of experience and salary.
The model predicts salary with reasonable accuracy using experience as the only input.
