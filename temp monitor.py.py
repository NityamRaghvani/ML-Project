import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. DATA GENERATION (Simulating Real-Life Sensors)
np.random.seed(42)
data_size = 500
cpu_load = np.random.randint(5, 95, data_size)
fan_speed = np.random.randint(1000,5000, data_size)

# Formula with noise: Higher CPU = Hotter, Higher Fan = Cooler
temp = (0.30 * cpu_load) - (0.004 * fan_speed) + 25 + np.random.normal(0, 1.5, data_size)

df = pd.DataFrame({'CPU_Load': cpu_load, 'Fan_RPM': fan_speed, 'Temp_C': temp})

# 2. DATA SPLITTING
X = df[['CPU_Load', 'Fan_RPM']]
y = df['Temp_C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 4. EVALUATION (The "Accuracy" Part)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("____ Model Performance ____")
print(f"Accuracy (R2 Score): {r2:.4f}")
print(f"Average Error (MAE): {mae:.2f}°C")
print("-------------------------\n")

# 5. Plotting Actual vs Predicted
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions, color='blue')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')

plt.title("How Accurate is our Model?")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.show()

# 6. REAL-WORLD TEST CASE
sample_input = np.array([[85,2000]])
predicted_alert = model.predict(sample_input)

print(f"\n[REAL-TIME TEST]")
print(f"Input: CPU 85%, Fan 2000 RPM")
print(f"Predicted Temperature: {predicted_alert[0]:.2f}°C")

if predicted_alert[0] > 40:
    print("STATUS: ALERT! High temperature predicted. Immediate cooling required.")
else:
    print("STATUS: Temperature within safe operational limits.")

