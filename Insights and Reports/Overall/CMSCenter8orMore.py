import pandas as pd
import re

def generate_cms_center_report_min8(input_csv="Data/Sankey_Pull_March.csv"):
    """
    Reads the CSV and prints, for each Season (in order),
    each CMS Center's Enrollments and Completions counts,
    but ONLY for those Season+Center combos that have
    8 or more enrollments.

    Also excludes rows where Full_Cohort == 'yes' (case-insensitive).
    Recognizes "2021" as a valid Season in addition to the usual
    'Winter|Spring|Summer|Fall YYYY'.
    """

    # 1) Load data
    df = pd.read_csv(input_csv)

    # 2) Validate/normalize "Season"
    def validate_season(season):
        if pd.isna(season):
            return "Unknown Season"
        season_str = str(season).strip()
        if season_str == "2021":
            return "2021"
        pattern = r'^(Winter|Spring|Summer|Fall)\s\d{4}$'
        if re.match(pattern, season_str):
            return season_str
        else:
            return "Unknown Season"

    df["Season"] = df["Season"].apply(validate_season)

    # 3) Define the season order so we can iterate from earliest to latest
    season_order = [
        "2021",
        "Winter 2022", "Spring 2022", "Summer 2022", "Fall 2022",
        "Winter 2023", "Spring 2023", "Summer 2023", "Fall 2023",
        "Winter 2024", "Spring 2024", "Summer 2024", "Fall 2024",
        "Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025",
        "Unknown Season"
    ]

    # 4) Convert "Verified Complete" to boolean
    df["Verified Complete"] = (
        df["Verified Complete"]
        .astype(str)
        .str.lower()
        .map({"yes": True, "true": True, "1": True, "no": False, "false": False, "0": False})
        .fillna(False)
    )

    # 5) Exclude rows where Full_Cohort == 'yes' (case-insensitive)
    df["Full_Cohort"] = df["Full_Cohort"].fillna("").astype(str)
    df = df[df["Full_Cohort"].str.lower() != "yes"].copy()

    # 6) Compute Enrollments & Completions by (Season, CMS Center)
    grouped = df.groupby(["Season", "CMS Center"], dropna=False).agg(
        Enrollments=("Email", "count"),
        Completions=("Verified Complete", "sum")
    ).reset_index()

    # 7) Print results
    for season in season_order:
        # Filter to rows for this season AND where Enrollments >= 8
        subset = grouped[
            (grouped["Season"] == season) &
            (grouped["Enrollments"] >= 8)
        ]
        if subset.empty:
            continue  # No rows to print for this season

        print(f"=== Season: {season} ===")
        for _, row in subset.iterrows():
            center = row["CMS Center"]
            enroll = row["Enrollments"]
            compl  = row["Completions"]
            print(f"  {center}, Enrollments, {enroll}")
            print(f"  {center}, Completions, {compl}")

if __name__ == "__main__":
    generate_cms_center_report_min8("Data/Sankey_Pull_March.csv")
