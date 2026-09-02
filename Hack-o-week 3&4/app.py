from flask import Flask, render_template, request, redirect
from analyzer import StudentAnalyzer
import pandas as pd

app = Flask(__name__)

CSV_FILE = "students.csv"


@app.route("/")
def home():

    analyzer = StudentAnalyzer(CSV_FILE)
    data = analyzer.process()

    return render_template(
        "index.html",
        students=data["students"],
        top_students=data["top_students"],
        bottom_students=data["bottom_students"],
        subject_average=data["subject_average"],
        statistics=data["statistics"]
    )


@app.route("/add", methods=["POST"])
def add_student():

    new_student = {
        "RollNo": request.form["roll"],
        "Name": request.form["name"],
        "Math": int(request.form["math"]),
        "Science": int(request.form["science"]),
        "English": int(request.form["english"]),
        "Attendance": int(request.form["attendance"])
    }

    df = pd.read_csv(CSV_FILE)

    df.loc[len(df)] = new_student

    df.to_csv(CSV_FILE, index=False)

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)