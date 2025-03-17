import pandas as pd
import re

def generate_unique_vs_nonunique_report(
    input_csv="Data/Sankey_Pull_March.csv",
    output_html="unique_vs_nonunique_report.html"
):
    """
    For each Season (ignoring rows where Full_Cohort=='yes'):
      1) Total Registrations (all rows)
      2) Unique Learners (distinct Email in a season)
      3) Non-Unique Learners (the 2nd+ mention of the same Email in that season)
      4) Unique Registrations (distinct Email+Track in a season)
      5) Non-Unique Registrations (the 2nd+ mention of the same Email+Track in that season)

      Then for completions (where Verified Complete == yes):
      6) Total Completions
      7) Unique Learner Completions (distinct Email among completions)
      8) Non-Unique Learner Completions
      9) Unique Registration Completions (distinct Email+Track among completions)
      10) Non-Unique Registration Completions

    Outputs an HTML file with the table.
    """

    # 1) Read CSV
    df = pd.read_csv(input_csv)

    # 2) Exclude rows where Full_Cohort == 'yes'
    df["Full_Cohort"] = df["Full_Cohort"].fillna("").astype(str)
    df = df[df["Full_Cohort"].str.lower() != "yes"].copy()

    # 3) Normalize 'Season'
    def validate_season(season_val):
        if pd.isna(season_val):
            return "Unknown Season"
        season_str = str(season_val).strip()
        if season_str == "2021":
            return "2021"
        pattern = r'^(Winter|Spring|Summer|Fall)\s\d{4}$'
        if re.match(pattern, season_str):
            return season_str
        return "Unknown Season"

    df["Season"] = df["Season"].apply(validate_season)

    # 4) Mark 'Verified Complete' as True only if 'yes'
    df["Verified Complete"] = (
        df["Verified Complete"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("yes")
    )

    # For sorting: known season order
    season_order = [
        "2021",
        "Winter 2022", "Spring 2022", "Summer 2022", "Fall 2022",
        "Winter 2023", "Spring 2023", "Summer 2023", "Fall 2023",
        "Winter 2024", "Spring 2024", "Summer 2024", "Fall 2024",
        "Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025",
        "Unknown Season"
    ]

    # 5) Basic aggregator for total rows and completions
    season_agg = (
        df.groupby("Season", dropna=False)
          .agg(
              total_registrations=("Email", "size"),          # all rows
              total_completions=("Verified Complete", "sum")  # sum of True = number of completions
          )
          .reset_index()
    )

    # 6) Unique counts for entire registrations (not completed-only):
    #    a) Unique Learners: distinct Emails
    #    b) Unique Registrations: distinct (Email, Track)
    unique_learners_all = (
        df.drop_duplicates(["Season", "Email"])
          .groupby("Season")
          .size()
          .reset_index(name="unique_learners")
    )

    unique_registrations_all = (
        df.drop_duplicates(["Season", "Email", "Track"])
          .groupby("Season")
          .size()
          .reset_index(name="unique_registrations")
    )

    # 7) For completions only
    df_completions = df[df["Verified Complete"]].copy()

    #    a) Unique Learner Completions: distinct Email in completed rows
    unique_learner_completions = (
        df_completions.drop_duplicates(["Season", "Email"])
        .groupby("Season")
        .size()
        .reset_index(name="unique_learner_completions")
    )

    #    b) Unique Registration Completions: distinct (Email, Track) in completed rows
    unique_registration_completions = (
        df_completions.drop_duplicates(["Season", "Email", "Track"])
        .groupby("Season")
        .size()
        .reset_index(name="unique_registration_completions")
    )

    # 8) Merge everything into season_agg
    season_agg = season_agg.merge(unique_learners_all, on="Season", how="left")
    season_agg = season_agg.merge(unique_registrations_all, on="Season", how="left")
    season_agg = season_agg.merge(unique_learner_completions, on="Season", how="left")
    season_agg = season_agg.merge(unique_registration_completions, on="Season", how="left")

    # Fill NAs with 0
    fill_cols = [
        "unique_learners", 
        "unique_registrations",
        "unique_learner_completions",
        "unique_registration_completions"
    ]
    for c in fill_cols:
        season_agg[c] = season_agg[c].fillna(0).astype(int)

    # 9) Filter out seasons with zero total_registrations if desired
    season_agg = season_agg[season_agg["total_registrations"] > 0].copy()

    # 10) Convert Season to a categorical to sort in custom order
    season_agg["Season"] = pd.Categorical(season_agg["Season"], categories=season_order, ordered=True)
    season_agg.sort_values("Season", inplace=True)

    # 11) Compute non-unique columns
    #     - Non-Unique Learners = total_registrations - unique_learners
    #     - Non-Unique Registrations = total_registrations - unique_registrations
    #     - Non-Unique Learner Completions = total_completions - unique_learner_completions
    #     - Non-Unique Registration Completions = total_completions - unique_registration_completions
    season_agg["non_unique_learners"] = season_agg["total_registrations"] - season_agg["unique_learners"]
    season_agg["non_unique_registrations"] = season_agg["total_registrations"] - season_agg["unique_registrations"]
    season_agg["non_unique_learner_completions"] = season_agg["total_completions"] - season_agg["unique_learner_completions"]
    season_agg["non_unique_registration_completions"] = season_agg["total_completions"] - season_agg["unique_registration_completions"]

    # 12) Rename columns for final output
    season_agg.rename(columns={
        "total_registrations": "Total Registrations",
        "unique_learners": "Unique Learners",
        "non_unique_learners": "Non-Unique Learners",
        "unique_registrations": "Unique Registrations",
        "non_unique_registrations": "Non-Unique Registrations",
        "total_completions": "Total Completions",
        "unique_learner_completions": "Unique Learner Completions",
        "non_unique_learner_completions": "Non-Unique Learner Completions",
        "unique_registration_completions": "Unique Registration Completions",
        "non_unique_registration_completions": "Non-Unique Registration Completions",
    }, inplace=True)

    # 13) Choose final column order
    final_cols = [
        "Season",
        "Total Registrations",
        "Unique Learners",
        "Non-Unique Learners",
        "Unique Registrations",
        "Non-Unique Registrations",
        "Total Completions",
        "Unique Learner Completions",
        "Non-Unique Learner Completions",
        "Unique Registration Completions",
        "Non-Unique Registration Completions"
    ]
    season_agg = season_agg[final_cols]

    # 14) Convert to HTML
    html_table = season_agg.to_html(index=False)

    # 15) Write to file
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Unique vs Non-Unique</title></head><body>\n")
        f.write("<h1>Registration and Completion Counts by Season</h1>\n")
        f.write(html_table)
        f.write("</body></html>\n")

    print(f"Report generated: {output_html}")
    print("Open the HTML file to view. Then copy/paste the table into Word as needed.")

if __name__ == "__main__":
    generate_unique_vs_nonunique_report(
        input_csv="Data/Sankey_Pull_March.csv",
        output_html="unique_vs_nonunique_report.html"
    )
