import pandas as pd
import numpy as np
import os

# Fix randomness
np.random.seed(42)

data = []

for i in range(200):
    attendance = np.random.randint(60, 100)
    tasks = np.random.randint(3, 15)
    coding = np.random.randint(40, 100)
    communication = np.random.randint(40, 100)
    deadline = np.random.choice(["Yes", "No"])
    mentor = np.random.randint(50, 100)

    score = (attendance + coding + communication + mentor) / 4

    if score > 75:
        performance = "Good"
    elif score > 60:
        performance = "Average"
    else:
        performance = "Poor"

    data.append([
        i+1, attendance, tasks, coding,
        communication, deadline, mentor, performance
    ])

df = pd.DataFrame(data, columns=[
    "Intern_ID", "Attendance", "Tasks_Completed",
    "Coding_Score", "Communication_Rating",
    "Deadline_Met", "Mentor_Feedback", "Performance_Label"
])

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/intern_performance_dataset.csv", index=False)

print("Dataset created successfully!")