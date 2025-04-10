import random
import pandas as pd

def generate_sample_data(n=500):
    data = {
        'Age': [random.randint(18, 65) for _ in range(n)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(n)],
        'Height_cm': [random.randint(150, 200) for _ in range(n)],
        'Weight_kg': [random.randint(50, 120) for _ in range(n)],
        'Steps': [random.randint(3000, 15000) for _ in range(n)],
        'Sleep_hours': [round(random.uniform(4, 10), 1) for _ in range(n)],
        'Activity_level': [random.choice(['Low', 'Moderate', 'High']) for _ in range(n)],
        'Calories': [random.randint(1500, 3500) for _ in range(n)]
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = generate_sample_data()
    df.to_csv("iot_calorie_data.csv", index=False)