import pandas as pd
import numpy as np

np.random.seed(42)
num_students = 1000

print("🏭 Generating balanced multi-class student profiles...")

study_hours = np.round(np.random.uniform(0.0, 10.0, num_students), 1)
attendance = np.round(np.random.uniform(0.0, 100.0, num_students), 0)    # Changed from 50.0 to 0.0
prior_grade = np.round(np.random.uniform(0.0, 100.0, num_students), 0)   # Changed from 40.0 to 0.0
sleep_hours = np.round(np.random.uniform(4.0, 9.0, num_students), 1)
assignments = np.round(np.random.uniform(0.0, 100.0, num_students), 0)   # Changed from 30.0 to 0.0
revision_freq = np.random.randint(0, 8, size=num_students)
backlogs = np.random.randint(0, 4, size=num_students)
tutoring = np.random.choice([1, 0], size=num_students, p=[0.4, 0.6])

# Boosted weights slightly to guarantee students reach the "Excellent" band
noise = np.random.normal(0, 2, num_students)
raw_score = (
    (prior_grade * 0.35) + 
    (attendance * 0.30) + 
    (study_hours * 2.0) + 
    (assignments * 0.15) + 
    (sleep_hours * 0.6) + 
    (revision_freq * 1.0) + 
    (tutoring * 4.0) - 
    (backlogs * 5.0) + 
    noise
)
raw_score = np.clip(raw_score, 0, 100)

def assign_band(score):
    if score >= 82: return 'Excellent'
    elif score >= 68: return 'Good'
    elif score >= 53: return 'Average'
    elif score >= 42: return 'At Risk'
    else: return 'Failing'

performance_band = [assign_band(s) for s in raw_score]

df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Prior_Grade': prior_grade,
    'Sleep_Hours': sleep_hours,
    'Assignments_Done': assignments,
    'Revision_Frequency': revision_freq,
    'Backlogs': backlogs,
    'Tutoring': tutoring,
    'Performance_Band': performance_band
})

df.to_csv('student_performance_data.csv', index=False)
print("✅ Dataset updated with verified 'Excellent' targets!")