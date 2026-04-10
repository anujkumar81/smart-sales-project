import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("data.csv")

# Data cleaning
df.dropna(inplace=True)

# Show data
print("Dataset:\n", df)

# Visualization
plt.plot(df['Month'], df['Sales'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Machine Learning Model
X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

# Prediction for next month (Month 11)
prediction = model.predict([[11]])

print("\n📈 Predicted Sales for Next Month:", int(prediction[0]))