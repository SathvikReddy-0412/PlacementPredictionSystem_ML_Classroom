import os

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_data import load_data


sns.set_style("whitegrid")

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "placement_predict_50k Dataset.csv"
)

def run_eda() -> dict:
    data = load_data(CSV_PATH)

    charts = []

    #3. MISSING VALUES
    print("\n"+"="*80)
    print("3.MISSING VALUES")
    print("="*80)
    missing=data.isnull().sum()
    missing_pct=(missing / len(data))*100
    missing_df=pd.DataFrame({"missing_count":missing,"missing_pct":missing_pct})
    missing_df=missing_df[missing_df["missing_pct"]>0].sort_values(by="missing_count",ascending=False)
    print(missing_df)


    if not missing_df.empty:
        plt.figure(figsize=(10,5))
        sns.barplot(x=missing_df.index,y=missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing %")
        plt.title("Missing values by column")
        _save(_chart_path("missing_values.png"))
        charts.append("missing_values.png")

    #4. Duplicates
    print("\n" + "=" * 80)
    print("4.Duplicate Rows")
    print("=" * 80)
    print("Duplicate rows:", data.duplicated().sum())

    #5. Target variable distribution (placement status)

    target_counts = data["PlacementStatus"].value_counts().to_dict()
    plt.figure()
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("Placement Status (0 = Not placed, 1 = placed)")
    plt.ylabel("Count")
    plt.title("Placement Status Distribution")
    _save(_chart_path("placement_status.png"))
    charts.append("placement_status.png")

    #6. NUMERIC FEATURE DISTRIBUTIONS
    print("\n" + "=" * 80)
    print("6. NUMERIC DISTRIBUTION")
    print("=" * 80)

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "Publications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "ExtraCurricular",
        "CGPA_Tier",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]
    data[hist_cols].hist(figsize=(14, 10), bins=24)
    _save(_chart_path("numeric_distribution.png"))
    charts.append("numeric_distribution.png")

    # Mean Line example
    plt.figure(dpi=125)
    sns.histplot(data=data["CGPA"], kde=True)
    plt.axvline(np.mean(data["CGPA"]), color="green", linestyle="--", label="Mean")
    plt.legend()
    plt.title("CGPA Distribution with Mean")
    _save(_chart_path("cgpa_distribution.png"))
    charts.append("cgpa_distribution.png")

    #7. OUTLIER DETECTION(DBXPLOTS)
    print("\n" + "=" * 80)
    print("7. OUTLIER DETECTION")
    print("=" * 80)

    box_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "Workshops", "Certifications", "Publications",
                 "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
                 "CGPA_Tier", "PlacementStatus", "IsAnomaly", "Salary Package"]
    box_cols = [c for c in box_cols if c in data.columns]
    for col in box_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=data[col], color="skyblue")
        plt.title(f"Boxplot = {col}", fontsize=14)
        plt.show()




    #8. CORRELATION HEATMAP
    print("\n" + "=" * 80)
    print("8.CORRELERATION ANALYSIS")
    print("=" * 80)
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(16, 12), dpi=100)
    sns.heatmap(np.round(corr, 2), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()





    # 9. RELATIONSHIP PLOTS

    print("\n" + "=" * 80)
    print("9. RELATIONSHIP PLOTS")
    print("=" * 80)
    if {
        "CGPA",
        "Salary Package"
    }.issubset(data.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(
            data=data,
            x="CGPA",
            y="Salary Package"
        )
        plt.title(
            "CGPA vs Salary Package"
        )
        plt.show()
    if {
        "AptitudeTestScore",
        "CodingTestScore"
    }.issubset(data.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(
            data=data,
            x="AptitudeTestScore",
            y="CodingTestScore"
        )
        plt.title(
            "Aptitude vs Coding Test Score"
        )
        plt.show()


    # 10. CATEGORICAL FEATURE COUNTS

    print("\n" + "=" * 80)
    print("10. CATEGORICAL COUNTS")
    print("=" * 80)
    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]
    cat_cols = [
        c for c in cat_cols
        if c in data.columns
    ]
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(
            data=data,
            x=col
        )
        plt.xticks(
            rotation=45
        )
        plt.title(
            f"{col} Distribution"
        )
        plt.show()


    # 11. GENDER VS PLACEMENT STATUS

    print("\n" + "=" * 80)
    print("11. GENDER VS PLACEMENT STATUS")
    print("=" * 80)
    if {
        "Gender",
        "PlacementStatus"
    }.issubset(data.columns):
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=data,
            x="Gender",
            hue="PlacementStatus"
        )
        plt.title(
            "Gender vs Placement Status"
        )
        plt.show()



    # 12. COLLEGE TIER / STREAM VS PLACEMENT

    print("\n" + "=" * 80)
    print("12. COLLEGE TIER / STREAM VS PLACEMENT")
    print("=" * 80)
    if {
        "CollegeTier",
        "PlacementStatus"
    }.issubset(data.columns):
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=data,
            x="CollegeTier",
            hue="PlacementStatus"
        )
        plt.title(
            "College Tier vs Placement Status"
        )
        plt.show()
    if {
        "Stream",
        "PlacementStatus"
    }.issubset(data.columns):
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=data,
            x="Stream",
            hue="PlacementStatus"
        )
        plt.xticks(
            rotation=45
        )
        plt.title(
            "Stream vs Placement Status"
        )
        plt.show()

    # 13. SGPA TREND ACROSS SEMESTERS

    print("\n" + "=" * 80)
    print("13. SGPA TREND")
    print("=" * 80)
    sgpa_cols = [
        "Sem1_SGPA",
        "Sem2_SGPA",
        "Sem3_SGPA",
        "Sem4_SGPA",
        "Sem5_SGPA",
        "Sem6_SGPA",
        "Sem7_SGPA",
        "Sem8_SGPA"
    ]
    sgpa_cols = [
        c for c in sgpa_cols
        if c in data.columns
    ]
    if len(sgpa_cols) > 0:
        avg_sgpa = (
            data[sgpa_cols]
            .mean()
        )
        plt.figure(figsize=(8, 5))
        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o"
        )
        plt.xticks(
            rotation=45
        )
        plt.xlabel(
            "Semester"
        )
        plt.ylabel(
            "Average SGPA"
        )
        plt.title(
            "Average SGPA Trend Across Semesters"
        )
        plt.show()

    # 14. SALARY PACKAGE ANALYSIS

    print("\n" + "=" * 80)
    print("14. SALARY ANALYSIS")
    print("=" * 80)
    if {
        "Salary Package",
        "PlacementStatus"
    }.issubset(data.columns):
        placed_students = data[
            data["PlacementStatus"] == 1
            ]
        plt.figure(figsize=(8, 5))
        sns.histplot(
            placed_students["Salary Package"],
            kde=True
        )
        plt.title(
            "Salary Distribution for Placed Students"
        )
        plt.show()
    if {
        "Salary Package",
        "CollegeTier"
    }.issubset(data.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=data,
            x="CollegeTier",
            y="Salary Package"
        )
        plt.title(
            "Salary Package by College Tier"
        )
        plt.show()

    # 15. PAIRPLOT

    print("\n" + "=" * 80)
    print("15. PAIRPLOT")
    print("=" * 80)
    pair_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "PlacementStatus"
    ]
    pair_cols = [
        c for c in pair_cols
        if c in data.columns
    ]
    if len(pair_cols) == 5:
        sns.pairplot(
            data[pair_cols],
            hue="PlacementStatus"
        )
        plt.show()



    return {
        "charts": charts
    }


if __name__ == "__main__":
    result = run_eda()
    print(result)
