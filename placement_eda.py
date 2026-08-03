import os
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
    plt.savefig(filename, bbox_inches="tight",dpi=100)
    plt.close()


CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "placement_predict_50k Dataset.csv"
)

def run_eda() -> dict:
    data = load_data(CSV_PATH)

    charts = []


    #MISSING VALUES
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

    #Duplicates
    print("\n" + "=" * 80)
    print("4.Duplicate Rows")
    print("=" * 80)
    print("Duplicate rows:", data.duplicated().sum())

    # Target variable distribution (placement status)

    target_counts = data["PlacementStatus"].value_counts().to_dict()
    plt.figure()
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("Placement Status (0 = Not placed, 1 = placed)")
    plt.ylabel("Count")
    plt.title("Placement Status Distribution")
    _save(_chart_path("placement_status.png"))
    charts.append("placement_status.png")



if __name__ == "__main__":
    result = run_eda()
    print(result)
