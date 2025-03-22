import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Air_Quality.csv")
features = ['Geo Place Name', 'Time Period', 'Measure', 'Indicator ID']
target = 'Data Value'
df = df[features + [target]].dropna()
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_features = ['Geo Place Name', 'Time Period', 'Measure']
numerical_features = ['Indicator ID']

transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ])
model = Pipeline(steps=[('preprocessor', transformer),
                        ('regressor', LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Air Quality Values")
plt.show()
new_data = pd.DataFrame({
    'Geo Place Name': ['Upper East Side-Gramercy'],
    'Time Period': ['Summer 2014'],
    'Measure': ['Mean'],
    'Indicator ID': [386]
})
predicted_value = model.predict(new_data)
print(f'Predicted Air Quality Value: {predicted_value[0]}')
