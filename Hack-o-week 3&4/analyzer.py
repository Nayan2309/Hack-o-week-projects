import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class StudentAnalyzer:

    def __init__(self, file_name):
        self.file_name = file_name
        self.df = None

    # Load CSV File
    def load_data(self):
        self.df = pd.read_csv(self.file_name)
        return self.df

    # Clean Data
    def clean_data(self):
        self.df.drop_duplicates(inplace=True)
        self.df.fillna(0, inplace=True)
        return self.df

    # Calculate Total Marks
    def calculate_total(self):
        self.df["Total"] = (
            self.df["Math"] +
            self.df["Science"] +
            self.df["English"]
        )
        return self.df

    # Calculate Percentage
    def calculate_percentage(self):
        self.df["Percentage"] = (
            self.df["Total"] / 300 * 100
        ).round(2)
        return self.df

    # Assign Grades
    def assign_grade(self):

        grades = []

        for percentage in self.df["Percentage"]:

            if percentage >= 90:
                grades.append("A+")

            elif percentage >= 80:
                grades.append("A")

            elif percentage >= 70:
                grades.append("B")

            elif percentage >= 60:
                grades.append("C")

            elif percentage >= 50:
                grades.append("D")

            else:
                grades.append("F")

        self.df["Grade"] = grades

        return self.df
        # Calculate Rank
    def calculate_rank(self):
        self.df["Rank"] = self.df["Percentage"].rank(
            ascending=False,
            method="dense"
        ).astype(int)

        return self.df

    # Scholarship Eligibility
    def scholarship_status(self):

        status = []

        for percentage in self.df["Percentage"]:

            if percentage >= 85:
                status.append("Eligible")
            else:
                status.append("Not Eligible")

        self.df["Scholarship"] = status

        return self.df

    # Top 5 Students
    def top_students(self):
        return self.df.sort_values(
            by="Percentage",
            ascending=False
        ).head(5)

    # Bottom 5 Students
    def bottom_students(self):
        return self.df.sort_values(
            by="Percentage",
            ascending=True
        ).head(5)

    # Subject Average
    def subject_average(self):

        return {
            "Math": round(self.df["Math"].mean(), 2),
            "Science": round(self.df["Science"].mean(), 2),
            "English": round(self.df["English"].mean(), 2)
        }

    # Overall Statistics using NumPy
    def statistics(self):

        percentages = self.df["Percentage"].to_numpy()

        return {
            "Highest": np.max(percentages),
            "Lowest": np.min(percentages),
            "Average": round(np.mean(percentages), 2),
            "Median": round(np.median(percentages), 2),
            "Standard Deviation": round(np.std(percentages), 2)
        }
        # Create Graphs
    def create_graphs(self):

        # Bar Chart
        plt.figure(figsize=(8,5))
        sns.barplot(
            x="Name",
            y="Percentage",
            data=self.df,
            palette="viridis"
        )
        plt.title("Student Percentage")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("static/bar_chart.png")
        plt.close()

        # Pie Chart
        grade_count = self.df["Grade"].value_counts()

        plt.figure(figsize=(6,6))
        plt.pie(
            grade_count,
            labels=grade_count.index,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Grade Distribution")
        plt.savefig("static/pie_chart.png")
        plt.close()

        # Heatmap
        plt.figure(figsize=(6,4))
        sns.heatmap(
            self.df[["Math","Science","English"]].corr(),
            annot=True,
            cmap="Blues"
        )
        plt.title("Subject Correlation")
        plt.tight_layout()
        plt.savefig("static/heatmap.png")
        plt.close()

    # Process Complete Dataset
    def process(self):

        self.load_data()
        self.clean_data()
        self.calculate_total()
        self.calculate_percentage()
        self.assign_grade()
        self.calculate_rank()
        self.scholarship_status()
        self.create_graphs()

        return {
            "students": self.df.to_dict(orient="records"),
            "top_students": self.top_students().to_dict(orient="records"),
            "bottom_students": self.bottom_students().to_dict(orient="records"),
            "subject_average": self.subject_average(),
            "statistics": self.statistics()
        }