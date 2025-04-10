import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import joblib
from preprocessing import preprocess_data

X, y = preprocess_data("iot_calorie_data.csv")
model = joblib.load('calorie_predictor_model.pkl')
y_pred = model.predict(X)

plt.figure(figsize=(10, 5))
plt.plot(y.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted', linestyle='dashed')
plt.title('Calorie Prediction vs Actual')
plt.legend()
plt.show()