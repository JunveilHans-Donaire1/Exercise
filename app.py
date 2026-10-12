
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("student_pass_model.joblib")

st.title("Student Pass Prediction")
st.write("Predict whether a student will pass based on study hours.")


student_name = st.text_input("Enter Student Name")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = model.predict([[study_hours]])[0]

        st.subheader("Prediction Result")
        st.write(f"Student Name: {student_name}")
        st.write(f"Study Hours: {study_hours}")

        if prediction == "Pass":
            st.success(f"{student_name} is predicted to PASS!")
            st.write(
                "Explanation: Based on the training data, "
                "the model classifies these study hours as Pass."
            )
        else:
            st.error(f"{student_name} is predicted to FAIL.")
            st.write(
                "Explanation: Based on the training data, "
                "the model classifies these study hours as Fail."
            )

        st.subheader("Study Hours and Prediction Chart")

        chart_data = pd.DataFrame({
            "Study Hours": [1, 2, 3, 4, 5, 6, 7, 8],
            "Actual Result": [
                "Fail", "Fail", "Fail", "Pass",
                "Pass", "Pass", "Pass", "Pass"
            ]
        })

        chart_data["Result Number"] = chart_data[
            "Actual Result"
        ].map({"Fail": 0, "Pass": 1})

        fig, ax = plt.subplots()

        ax.plot(
            chart_data["Study Hours"],
            chart_data["Result Number"],
            marker="o",
            label="Training Data"
        )

        ax.scatter(
            [study_hours],
            [1 if prediction == "Pass" else 0],
            marker="*",
            s=200,
            label="Student Prediction"
        )

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Result")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fail", "Pass"])
        ax.set_title("Student Pass Prediction")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
