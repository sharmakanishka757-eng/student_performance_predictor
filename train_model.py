import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_csv('student_performance_data.csv')

# 2. Split features and targets
X = df[['Study_Hours', 'Attendance', 'Prior_Grade', 'Sleep_Hours', 
        'Assignments_Done', 'Revision_Frequency', 'Backlogs', 'Tutoring']]
y = df['Performance_Band']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest Classifier for 5 Multi-Class Bands...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"📊 Model Training Accuracy: {acc * 100:.2f}%")

# Save the trained model file (overwriting any old regression assets)
with open('student_predictor_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("💾 Saved engine file as 'student_predictor_model.pkl'")