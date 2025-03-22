# Predictive-Analysis-using-ML
**COMPANY**: CODTECH IT SOLUTIONS 
**NAME**: TANYA DEEP 
**INTERN ID**: CT08WBB 
**DOMAIN**: DATA ANALYSIS 
**DURATION**: 4 WEEKS 
**MENTOR**: NEELA SANTOSH

**DESCRIPTION**
This Python script constructs and evaluates a linear regression model to predict air quality values from the "Air_Quality.csv" dataset. It begins by loading the data into a Pandas DataFrame, selecting relevant features (Geo Place Name, Time Period, Measure, Indicator ID), and the target variable (Data Value), then removing any rows with missing values. The dataset is split into training and testing sets to evaluate the model's performance. Feature engineering is then performed using a `ColumnTransformer` within a `Pipeline`; categorical features are one-hot encoded to convert them into numerical representations, and the numerical "Indicator ID" is standardized. A `LinearRegression` model is then trained on the processed training data, and predictions are made on the test set. The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) score, which are printed to the console. A scatter plot is generated to visualize the relationship between the actual and predicted air quality values, providing a visual assessment of the model's accuracy. Finally, the script demonstrates how to use the trained model to predict air quality for new, unseen data, showcasing the model's practical application.

* **Data Preprocessing:** Handles categorical and numerical features using `OneHotEncoder` and `StandardScaler` within a `ColumnTransformer`.
* **Model Training and Evaluation:** Trains a `LinearRegression` model and assesses its performance using MSE and R2 scores.
* **Visualization:** Generates a scatter plot to compare actual versus predicted values.
* **Prediction on New Data:** Demonstrates how to use the trained model for making predictions on new, unseen data.

**OUTPUT**
Mean Squared Error: 324.3631346814512
R-squared Score: 0.3924578780858481
Predicted Air Quality Value: 32.5925029063729
![Image](https://github.com/user-attachments/assets/3185bc4d-1f40-49c4-a71b-ee8066c1fea7)
