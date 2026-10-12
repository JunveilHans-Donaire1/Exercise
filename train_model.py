
import joblib
from sklearn.tree import DecisionTreeClassifier

# Example training data: study hours and pass/fail result
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully!")


study_hours = [[2], [5], [8]]
predictions = loaded_model.predict(study_hours)

for hours, prediction in zip(study_hours, predictions):
    print(f"Study hours: {hours[0]} - Prediction: {prediction}")
