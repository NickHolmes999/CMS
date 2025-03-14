import pandas as pd
import re
import math

def generate_aggregate_report(input_csv="Data/Sankey_Pull_March.csv", output_html="aggregate_report.html"):
    """
    Reads CSV data, excludes Full_Cohort=='yes', groups by Season (ignoring CMS centers).
    Produces two HTML tables (season-level, year-level) with counts, rates, and diffs,
    which you can open in a browser or copy/paste into Word.
    """

    # 1) LOAD DATA
    df = pd.read_csv(input_csv)

    # 2) EXCLUDE FULL_COHORT=='yes' (case-insensitive)
    df["Full_Cohort"] = df["Full_Cohort"].fillna("").astype(str)
    df = df[df["Full_Cohort"].str.lower() != "yes"].copy()

    # 3) VALIDATE/NORMALIZE "Season"
    def validate_season(season_val):
        if pd.isna(season_val):
            return "Unknown Season"
        season_str = str(season_val).strip()
        # Accept '2021' as a valid "season"
        if season_str == "2021":
            return "2021"
        # Also accept (Winter|Spring|Summer|Fall YYYY)
        pattern = r'^(Winter|Spring|Summer|Fall)\s\d{4}$'
        if re.match(pattern, season_str):
            return season_str
        return "Unknown Season"

    df["Season"] = df["Season"].apply(validate_season)

    # 4) TREAT 'VERIFIED COMPLETE' & 'DROPPED' AS TRUE ONLY IF == 'YES'
    df["Verified Complete"] = (
        df["Verified Complete"].fillna("").astype(str).str.lower().eq("yes")
    )
    df["Dropped"] = (
        df["Dropped"].fillna("").astype(str).str.lower().eq("yes")
    )

    # 5) DEFINE SEASON ORDER
    season_order = [
        "2021",
        "Winter 2022", "Spring 2022", "Summer 2022", "Fall 2022",
        "Winter 2023", "Spring 2023", "Summer 2023", "Fall 2023",
        "Winter 2024", "Spring 2024", "Summer 2024", "Fall 2024",
        "Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025",
        "Unknown Season"
    ]

    # 6) AGGREGATE BY SEASON
    ag_season = (
        df.groupby("Season", dropna=False)
          .agg(
              Enrollments=("Email", "count"),
              Completions=("Verified Complete", "sum"),
              Drops=("Dropped", "sum")
          )
          .reset_index()
    )

    # Remove seasons with 0 enrollments
    ag_season = ag_season[ag_season["Enrollments"] > 0].copy()

    # Put in known order
    ag_season["Season"] = pd.Categorical(
        ag_season["Season"], categories=season_order, ordered=True
    )
    ag_season.sort_values("Season", inplace=True)

    # 7) ADD RATES & DIFFS (SEASON-TO-SEASON)
    def add_rates_and_diffs(df_in, label_col="Season"):
        df_out = df_in.copy()
        # Compute completion & drop rates
        df_out["CompletionRate"] = df_out["Completions"] / df_out["Enrollments"]
        df_out["DropRate"]       = df_out["Drops"]       / df_out["Enrollments"]

        # Numeric diffs vs previous row
        df_out["Enrollments_diff"] = df_out["Enrollments"].diff()
        df_out["Completions_diff"] = df_out["Completions"].diff()
        df_out["Drops_diff"]       = df_out["Drops"].diff()

        # % diffs vs previous row's value
        df_out["Enrollments_diff_pct"] = (
            df_out["Enrollments_diff"] / df_out["Enrollments"].shift(1) * 100
        )
        df_out["Completions_diff_pct"] = (
            df_out["Completions_diff"] / df_out["Completions"].shift(1) * 100
        )
        df_out["Drops_diff_pct"] = (
            df_out["Drops_diff"] / df_out["Drops"].shift(1) * 100
        )

        # Replace NaN in the first row with 0
        for c in [
            "Enrollments_diff", "Completions_diff", "Drops_diff",
            "Enrollments_diff_pct", "Completions_diff_pct", "Drops_diff_pct"
        ]:
            df_out[c] = df_out[c].fillna(0)

        return df_out

    ag_season = add_rates_and_diffs(ag_season)

    # 8) AGGREGATE BY YEAR (extract from Season string)
    def parse_year(season_str):
        if season_str == "2021":
            return 2021
        match = re.search(r'(\d{4})$', str(season_str))
        if match:
            return int(match.group(1))
        return 0  # for "Unknown Season"

    ag_season["Year"] = ag_season["Season"].astype(str).apply(parse_year)

    ag_year = (
        ag_season.groupby("Year", dropna=False)
                 .agg(
                     Enrollments=("Enrollments", "sum"),
                     Completions=("Completions", "sum"),
                     Drops=("Drops", "sum")
                 )
                 .reset_index()
    )
    # Exclude unknown year=0
    ag_year = ag_year[ag_year["Year"] > 0].copy()
    ag_year.sort_values("Year", inplace=True)

    ag_year = add_rates_and_diffs(ag_year, label_col="Year")

    # 9) PREPARE TABLES AS DATAFRAMES FOR HTML EXPORT

    # A) SEASON-LEVEL
    df_season = ag_season.copy()
    df_season["Enroll"]               = df_season["Enrollments"]
    df_season["Enroll Diff"]          = df_season["Enrollments_diff"]
    df_season["Enroll Diff %"]        = df_season["Enrollments_diff_pct"]
    df_season["Complete"]             = df_season["Completions"]
    df_season["Comp Diff"]            = df_season["Completions_diff"]
    df_season["Comp Diff %"]          = df_season["Completions_diff_pct"]
    df_season["Drops Diff"]           = df_season["Drops_diff"]
    df_season["Drops Diff %"]         = df_season["Drops_diff_pct"]
    # Multiply completion rate and drop rate by 100
    df_season["Comp Rate"]            = df_season["CompletionRate"] * 100
    df_season["Drop Rate"]            = df_season["DropRate"] * 100

    season_table = df_season[
        [
            "Season", "Enroll", "Enroll Diff", "Enroll Diff %",
            "Complete", "Comp Diff", "Comp Diff %",
            "Drops", "Drops Diff", "Drops Diff %",
            "Comp Rate", "Drop Rate"
        ]
    ].copy()

    # B) YEAR-LEVEL
    df_year = ag_year.copy()
    df_year["Enroll"]               = df_year["Enrollments"]
    df_year["Enroll Diff"]          = df_year["Enrollments_diff"]
    df_year["Enroll Diff %"]        = df_year["Enrollments_diff_pct"]
    df_year["Complete"]             = df_year["Completions"]
    df_year["Comp Diff"]            = df_year["Completions_diff"]
    df_year["Comp Diff %"]          = df_year["Completions_diff_pct"]
    df_year["Drops Diff"]           = df_year["Drops_diff"]
    df_year["Drops Diff %"]         = df_year["Drops_diff_pct"]
    df_year["Comp Rate"]            = (df_year["Completions"] / df_year["Enrollments"] * 100).replace([math.inf, math.nan], 0)
    df_year["Drop Rate"]            = (df_year["Drops"] / df_year["Enrollments"] * 100).replace([math.inf, math.nan], 0)

    year_table = df_year[
        [
            "Year", "Enroll", "Enroll Diff", "Enroll Diff %",
            "Complete", "Comp Diff", "Comp Diff %",
            "Drops", "Drops Diff", "Drops Diff %",
            "Comp Rate", "Drop Rate"
        ]
    ].copy()

    # 10) EXPORT TO HTML
    season_html = season_table.to_html(index=False, float_format="%.2f")
    year_html   = year_table.to_html(index=False, float_format="%.2f")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Aggregate Report</title></head><body>\n")
        f.write("<h1>Season-Level Statistics</h1>\n")
        f.write(season_html)
        f.write("<br><hr>\n")
        f.write("<h1>Year-Level Statistics</h1>\n")
        f.write(year_html)
        f.write("</body></html>\n")

    print(f"HTML report generated: {output_html}")
    print("Open the file in your browser or Word. You can also copy/paste the tables into Word.")

if __name__ == "__main__":
    generate_aggregate_report(
        input_csv="Data/Sankey_Pull_March.csv",
        output_html="aggregate_report.html"
    )
