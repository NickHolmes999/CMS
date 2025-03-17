import pandas as pd
import re

def generate_center_single_pivot(input_csv="Data/Sankey_Pull_March.csv", output_html="center_single_pivot.html"):
    """
    Creates ONE pivot table with:
      - Rows = CMS Center
      - Columns = (Season, [Enrollments, Completions]) as a multi-level column
      - Values = numeric counts
    Then exports to an HTML file for easy copy/paste into Word.

    Excludes rows where Full_Cohort=='yes' (case-insensitive).
    Normalizes 'Season' so '2021' or 'Winter|Spring|Summer|Fall YYYY' remain,
    anything else => 'Unknown Season'.
    'Verified Complete' is True iff it equals 'yes' (case-insensitive).
    """

    # 1) LOAD CSV
    df = pd.read_csv(input_csv)

    # 2) EXCLUDE FULL_COHORT == 'yes'
    df["Full_Cohort"] = df["Full_Cohort"].fillna("").astype(str)
    df = df[df["Full_Cohort"].str.lower() != "yes"].copy()

    # 3) VALIDATE/NORMALIZE 'Season'
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

    # 4) CONVERT 'Verified Complete' to True only if == 'yes'
    df["Verified Complete"] = (
        df["Verified Complete"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("yes")
    )

    # 5) DEFINE A SEASON ORDER
    season_order = [
        "2021",
        "Winter 2022", "Spring 2022", "Summer 2022", "Fall 2022",
        "Winter 2023", "Spring 2023", "Summer 2023", "Fall 2023",
        "Winter 2024", "Spring 2024", "Summer 2024", "Fall 2024",
        "Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025",
        "Unknown Season"
    ]

    # 6) GROUP BY (CMS Center, Season) to get Enrollments & Completions
    grouped = (
        df.groupby(["CMS Center", "Season"], dropna=False)
          .agg(
              Enrollments=("Email", "count"),
              Completions=("Verified Complete", "sum")
          )
          .reset_index()
    )

    # 7) MELT (unpivot) from wide => tall, so we can pivot into a multi-level columns
    #    We'll get a "variable" column that is "Enrollments" or "Completions",
    #    and a "value" column with the numeric count.
    melted = grouped.melt(
        id_vars=["CMS Center", "Season"],
        value_vars=["Enrollments", "Completions"],
        var_name="Metric",
        value_name="Count"
    )

    # 8) BUILD A SINGLE PIVOT TABLE
    #    - index = CMS Center
    #    - columns = Season (level 1), Metric (level 2)
    #    - values = Count
    pivot_df = melted.pivot(
        index="CMS Center",
        columns=["Season", "Metric"],
        values="Count"
    ).fillna(0)

    # 9) SORT THE TOP-LEVEL COLUMNS BY OUR season_order
    #    The pivot has a MultiIndex in columns: (Season, Metric).
    #    We'll reorder the Season level using season_order.
    #    Then within each season, it will keep "Completions" & "Enrollments"
    #    in alphabetical order (or we can force an order). Let's force it:
    #    We want columns in the order: Season ascending by season_order,
    #    with "Enrollments" first, "Completions" second.
    #    We'll define a custom method for sorting the columns.

    # Current columns are something like: MultiIndex([
    #   ('2021', 'Enrollments'), ('2021', 'Completions'),
    #   ('Winter 2022', 'Enrollments'), ('Winter 2022', 'Completions'), ...
    # ])
    # We'll reorder them so that for each Season in season_order, we show
    # (Enrollments, Completions) in that order.

    def columns_in_order(pivot_df, season_order, second_level_order=["Enrollments", "Completions"]):
        # pivot_df.columns is a MultiIndex: [ (season, metric), ... ]
        # We'll build a new list of columns in the desired order.
        col_list = []
        # For each season in season_order
        for season in season_order:
            # For each metric in second_level_order
            for metric in second_level_order:
                if (season, metric) in pivot_df.columns:
                    col_list.append((season, metric))
        return col_list

    new_cols = columns_in_order(pivot_df, season_order, ["Enrollments", "Completions"])
    pivot_df = pivot_df[new_cols]

    # 10) EXPORT TO HTML
    html_table = pivot_df.to_html(index=True, float_format="%.0f")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Single CMS Center Pivot</title></head><body>\n")
        f.write("<h1>Single Pivot Table: (CMS Center) x (Season, Enrollments+Completions)</h1>\n")
        f.write(html_table)
        f.write("</body></html>\n")

    print(f"Single pivot table generated in '{output_html}'.")
    print("Open it to view or copy/paste the table into Word.")

if __name__ == "__main__":
    generate_center_single_pivot(
        input_csv="Data/Sankey_Pull_March.csv",
        output_html="center_single_pivot.html"
    )
