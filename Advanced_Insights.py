import pandas as pd
import numpy as np
import os
import re
import json
from collections import defaultdict

# Silence future fillna downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

def generate_insights_report(input_csv="Data/WR Sankey  copy.csv", output_html="report.html"):
    """
    Generates an HTML report with:
      - Follow-up Rate logic
      - By Track aggregator
      - Combined Track+Season+Level aggregator
      - Motivations, Pillars, etc.
      - Repeated enrollments (with times completed in track)
      - Track+Season+Offering aggregator, plus an Overall-by-Offering table.
      - Removes 'Unknown Offering' from offering tables, and hides zero-enrollment rows.
    """

    # 1) LOAD & VALIDATE
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' does not exist.")
    df = pd.read_csv(input_csv)

    required_cols = [
        "CMS Center Selection", "OIT Group", "Dropped", "Verified Complete",
        "Season Selection", "Track Selection", "Session Selection", "Email",
        "Pillar", "Offering", "Motivation", "Motivation_9_TEXT", "Level",
        "Special Interests", "CMS Group", "CMS Division", "BA Familiarity",
        "Benefits of Agile", "Applications of Agile", "Topic Interest",
        "Track Familiarity", "Benefits of Track", "Track Application",
        "Track Motivation", "Track Motivation_9_TEXT"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Validate / normalize Season
    def validate_season(season):
        if pd.isna(season):
            return "Unknown Season"
        pattern = r'^(Winter|Spring|Summer|Fall)\s\d{4}$'
        if re.match(pattern, season):
            return season
        else:
            return "Unknown Season"

    df["Season Selection"] = df["Season Selection"].apply(validate_season)

    # We define a fixed season_order
    season_order = [
        "Fall 2022", "Winter 2023", "Spring 2023", "Summer 2023",
        "Fall 2023", "Winter 2024", "Spring 2024", "Summer 2024",
        "Fall 2024", "Winter 2025", "Unknown Season"
    ]
    df["Season Selection"] = pd.Categorical(df["Season Selection"], categories=season_order, ordered=True)

    # Convert track to categorical
    df["Track Selection"] = df["Track Selection"].fillna("Unknown").astype(str)
    df["Track Selection"] = df["Track Selection"].replace(r'^nan$', "Unknown", regex=True)
    all_tracks_sorted = sorted(df["Track Selection"].unique())
    df["Track Selection"] = pd.Categorical(df["Track Selection"], categories=all_tracks_sorted, ordered=True)

    # Convert level
    df["Level"] = df["Level"].fillna("Unknown Level").astype(str)
    df["Level"] = df["Level"].replace(r'^nan$', "Unknown Level", regex=True)

    # Convert yes/no columns
    df["Dropped"] = df["Dropped"].str.lower().map({'yes': True, 'no': False}).fillna(False).astype(bool)
    df["Verified Complete"] = df["Verified Complete"].str.lower().map({'yes': True, 'no': False}).fillna(False).astype(bool)

    # If "Offering" is missing or NaN, keep it as NaN for now—then we'll filter it out later if needed.
    df["Offering"] = df["Offering"].astype(str)

    # 2) Basic Stats
    total_registrations = len(df)
    total_unique_participants = df["Email"].nunique()
    total_dropped = df["Dropped"].sum()
    total_completed = df["Verified Complete"].sum()
    drop_rate = total_dropped / total_registrations if total_registrations else 0
    completion_rate = total_completed / total_registrations if total_registrations else 0

    enrollments_by_participant = df.groupby("Email", observed=False).agg(
        total_enrollments=("Track Selection", "count"),
        total_completions=("Verified Complete", "sum")
    ).reset_index()
    participants_completed = enrollments_by_participant[enrollments_by_participant["total_completions"] > 0]
    num_completed = len(participants_completed)
    participants_followed_up = participants_completed[participants_completed["total_enrollments"] > 1]
    num_followed_up = len(participants_followed_up)
    overall_follow_up_rate = num_followed_up / num_completed if num_completed else 0

    # For combined tracks
    cyber_list = ["Cyber-Hygiene: Advanced Topics", "Cyber-Hygiene: Essentials"]
    ai_list = [
        "AI/ML: AI Applications in Government",
        "AI/ML: Applied Data Science Methods",
        "Artificial Intelligence and Machine Learning",
        "Data Science"
    ]

    # Helper to add combined track rows
    def add_combined_track_rows(original_df, group_cols, agg_dict, rename_dict=None,
                                compute_rates=None, include_unique_participants=False):
        grouped = original_df.groupby(group_cols, observed=False).agg(agg_dict).reset_index()

        if include_unique_participants:
            gp_uniq = (
                original_df.groupby(group_cols, observed=False)["Email"].nunique()
                .reset_index().rename(columns={"Email": "unique_participants"})
            )
            grouped = grouped.merge(gp_uniq, on=group_cols, how="left")

        # If grouping by track, we add combined track rows for Cyber/AI
        if "Track Selection" in group_cols:
            non_track_cols = [c for c in group_cols if c != "Track Selection"]
            if len(non_track_cols) == 0:
                combos = [{}]
            else:
                combos = grouped[non_track_cols].drop_duplicates().to_dict("records")

            combined_rows = []

            for combo in combos:
                def sum_for_combined(tracks_list, combined_name):
                    mask = pd.Series([True]*len(original_df))
                    for k,v in combo.items():
                        mask &= (original_df[k] == v)
                    mask &= original_df["Track Selection"].isin(tracks_list)
                    sub = original_df[mask]
                    if sub.empty:
                        return None
                    row_dict = dict(combo)
                    row_dict["Track Selection"] = combined_name
                    for c_, f_ in agg_dict.items():
                        if c_ == "Email" and f_ == "count":
                            row_dict["Email"] = len(sub)
                        elif c_ in ["Verified Complete","Dropped"] and f_ == "sum":
                            row_dict[c_] = sub[c_].sum()
                    if include_unique_participants:
                        row_dict["unique_participants"] = sub["Email"].nunique()
                    return row_dict

                r_c = sum_for_combined(cyber_list, "Cyber Combined")
                if r_c:
                    combined_rows.append(r_c)
                r_a = sum_for_combined(ai_list, "AI Combined")
                if r_a:
                    combined_rows.append(r_a)

            df_comb = pd.DataFrame(combined_rows)
            if not df_comb.empty:
                merged = pd.concat([grouped, df_comb], ignore_index=True)
            else:
                merged = grouped
        else:
            merged = grouped

        if rename_dict:
            merged = merged.rename(columns=rename_dict)
        if compute_rates:
            merged = compute_rates(merged)
        return merged

    # 3) Build a mapping from season -> numeric index
    season2idx = {s: i for i, s in enumerate(season_order)}

    # 4) Calculate earliest completed season for each (Email, Track)
    df_comp = df[df["Verified Complete"] == True].copy()
    df_comp["season_idx"] = df_comp["Season Selection"].map(season2idx)

    earliest_comp = (
        df_comp.groupby(["Email", "Track Selection"], observed=False)["season_idx"]
        .min()
        .reset_index()
        .rename(columns={"Track Selection": "Track"})
    )

    # 5) We'll need all enrollments (not necessarily completed)
    df["season_idx"] = df["Season Selection"].map(season2idx)
    df_valid = df.dropna(subset=["season_idx"])

    df_enroll = df_valid[["Email", "Track Selection", "season_idx"]].copy()
    df_enroll.rename(columns={"Track Selection": "Track"}, inplace=True)

    user_enroll_map = defaultdict(list)
    for _, row in df_enroll.iterrows():
        user = row["Email"]
        track = row["Track"]
        sidx = row["season_idx"]
        user_enroll_map[user].append((track, sidx))

    # 6) "Follow-up Rate" function
    def new_follow_up_rate(track_name):
        subset = earliest_comp[earliest_comp["Track"] == track_name]
        denominator = len(subset)
        if denominator == 0:
            return "0.00%"

        count_followed_up = 0
        for _, row_ in subset.iterrows():
            email = row_["Email"]
            sT    = row_["season_idx"]

            enrolls = user_enroll_map[email]
            followed_up = any(
                (t_ != track_name) and (s_ > sT)
                for (t_, s_) in enrolls
            )
            if followed_up:
                count_followed_up += 1

        ratio = count_followed_up / denominator
        return f"{ratio:.2%}"

    # 7) By Track aggregator
    def compute_track_rates(df_):
        df_["completion_rate"] = df_["Total Completed"] / df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]   / df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x: "{:.2%}".format(x))
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x: "{:.2%}".format(x))
        return df_

    track_agg = {"Email": "count", "Verified Complete": "sum", "Dropped": "sum"}
    track_stats = add_combined_track_rows(
        original_df=df,
        group_cols=["Track Selection"],
        agg_dict=track_agg,
        rename_dict={
            "Email": "Total Enrolled",
            "Verified Complete": "Total Completed",
            "Dropped": "Total Dropped"
        },
        compute_rates=None,
        include_unique_participants=True
    )
    track_stats = compute_track_rates(track_stats)

    follow_up_rates = []
    for _, row in track_stats.iterrows():
        tname = row["Track Selection"]
        f = new_follow_up_rate(tname)
        follow_up_rates.append(f)
    track_stats["Follow-up Rate (Track)"] = follow_up_rates

    col_order_track = [
        "Track Selection", "Total Enrolled", "Total Completed", "Total Dropped",
        "completion_rate", "drop_rate", "unique_participants", "Follow-up Rate (Track)"
    ]
    track_stats = track_stats[col_order_track]

    # 9) Season + Track + Level aggregator
    def compute_season_track_rates(df_):
        df_["completion_rate"] = df_["Total Completed"]/df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]/df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x:"{:.2%}".format(x))
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x:"{:.2%}".format(x))
        return df_

    stl_agg= {"Email":"count","Verified Complete":"sum","Dropped":"sum"}
    season_track_level_stats= add_combined_track_rows(
        original_df=df,
        group_cols=["Season Selection","Track Selection","Level"],
        agg_dict=stl_agg,
        rename_dict={
            "Email":"Total Enrolled",
            "Verified Complete":"Total Completed",
            "Dropped":"Total Dropped"
        },
        compute_rates=None,
        include_unique_participants=True
    )

    def add_all_levels_rows(df_, group_by=["Track Selection"], level_col="Level"):
        new_rows = []
        g = df_.groupby(group_by, observed=False)
        for keys, subdf in g:
            if not isinstance(keys, tuple):
                keys = (keys,)
            lvls = subdf[level_col].unique()
            if len(lvls) <= 1:
                continue
            row_dict = {}
            for i, c_ in enumerate(group_by):
                row_dict[c_] = keys[i]
            row_dict[level_col] = "All Levels"
            for c_ in ["Total Enrolled","Total Completed","Total Dropped"]:
                if c_ in subdf.columns:
                    row_dict[c_] = subdf[c_].sum()
            if "unique_participants" in subdf.columns:
                row_dict["unique_participants"] = subdf["unique_participants"].sum()
            new_rows.append(row_dict)
        if not new_rows:
            return df_
        df_new = pd.DataFrame(new_rows)
        out = pd.concat([df_, df_new], ignore_index=True)
        return out

    season_track_level_stats= add_all_levels_rows(
        season_track_level_stats, ["Season Selection","Track Selection"], "Level"
    )
    season_track_level_stats= compute_season_track_rates(season_track_level_stats)
    season_track_level_stats= season_track_level_stats.sort_values(
        by=["Track Selection","Level","Season Selection"],
        ascending=[True,True,True]
    ).reset_index(drop=True)
    season_track_level_stats["Total Enrolled"]= season_track_level_stats["Total Enrolled"].fillna(0)
    season_track_level_stats["Total Completed"]= season_track_level_stats["Total Completed"].fillna(0)
    season_track_level_stats["Total Dropped"]  = season_track_level_stats["Total Dropped"].fillna(0)

    # ----- Motivations (top 5 per track) -----
    def split_motivations(row):
        combined = ""
        if pd.notna(row["Motivation"]):
            combined += row["Motivation"]
        if pd.notna(row["Motivation_9_TEXT"]) and row["Motivation_9_TEXT"].strip() != "":
            if combined:
                combined += ", " + row["Motivation_9_TEXT"]
            else:
                combined = row["Motivation_9_TEXT"]
        items = [m.strip() for m in combined.split(",") if m.strip()]
        return items

    df["Motivation List"] = df.apply(split_motivations, axis=1)
    df_mot = df.explode("Motivation List")
    df_mot["Motivation List"] = df_mot["Motivation List"].str.strip()
    df_mot = df_mot[~df_mot["Motivation List"].str.lower().isin(["undefined","unknown","nan"])]
    df_mot = df_mot[df_mot["Motivation List"].str.len() >= 1]

    motivation_by_track = (
        df_mot.groupby(["Track Selection","Motivation List"], observed=False)
              .agg(Mentions=("Email","count"))
              .reset_index()
    )
    motivation_by_track = motivation_by_track.sort_values(["Track Selection","Mentions"], ascending=[True,False])
    motivation_by_track = motivation_by_track.groupby("Track Selection", observed=False).head(5).reset_index(drop=True)
    motivation_by_track = motivation_by_track.rename(columns={"Motivation List":"Motivation"})

    # Motivations by Pillar (Top 5)
    motivations_trends_by_pillar = (
        df_mot.groupby(["Pillar","Motivation List"], observed=False)
              .size()
              .reset_index(name="Count")
    )
    motivations_trends_by_pillar = motivations_trends_by_pillar.sort_values(["Pillar","Count"], ascending=[True,False])
    motivations_trends_by_pillar = motivations_trends_by_pillar.groupby("Pillar", observed=False).head(5).reset_index(drop=True)
    motivations_trends_by_pillar = motivations_trends_by_pillar.rename(columns={"Motivation List":"Motivation"})
    motivations_trends_by_pillar = motivations_trends_by_pillar[motivations_trends_by_pillar["Count"]!=0]

    # Topic Interest
    def rename_dist(d):
        d.rename(columns={"Email":"Count"}, inplace=True)
        return d

    topic_interest = add_combined_track_rows(
        original_df=df,
        group_cols=["Track Selection","Topic Interest"],
        agg_dict={"Email":"count"},
        rename_dict=None,
        compute_rates=rename_dist,
        include_unique_participants=False
    )
    topic_interest = topic_interest[topic_interest["Count"]!=0]
    topic_interest = topic_interest[~topic_interest["Topic Interest"].str.lower().isin(["undefined","unknown","nan"])]
    topic_interest = topic_interest.sort_values(["Track Selection","Topic Interest"]).reset_index(drop=True)

    # Pillar Insights
    def rename_pillar_cols(df_):
        df_.rename(columns={"Email":"Count"}, inplace=True)
        return df_

    pillar_by_track_agg = add_combined_track_rows(
        original_df=df,
        group_cols=["Track Selection","Pillar"],
        agg_dict={"Email":"count"},
        rename_dict=None,
        compute_rates=rename_pillar_cols,
        include_unique_participants=False
    )
    pillar_by_track_agg= pillar_by_track_agg.sort_values(["Track Selection","Pillar"], ascending=[True,True]).reset_index(drop=True)

    # BA Familiarity, Benefits, Applications => only for Business Agility
    def make_dist_table(col_):
        def rename_dist(dff):
            dff.rename(columns={"Email":"Count"}, inplace=True)
            return dff

        tmp = add_combined_track_rows(
            original_df=df,
            group_cols=["Track Selection", col_],
            agg_dict={"Email":"count"},
            rename_dict=None,
            compute_rates=rename_dist,
            include_unique_participants=False
        )
        # Only keep "Business Agility" track
        tmp = tmp[tmp["Track Selection"]=="Business Agility"]
        if "Track Selection" in tmp.columns:
            tmp.drop(columns=["Track Selection"], inplace=True)
        tmp= tmp.sort_values([col_], ascending=True).reset_index(drop=True)
        tmp= tmp[~tmp[col_].str.lower().isin(["undefined","unknown","nan"])]
        return tmp

    ba_familiarity_by_track = make_dist_table("BA Familiarity")
    benefits_agile_by_track = make_dist_table("Benefits of Agile")
    apps_agile_by_track     = make_dist_table("Applications of Agile")

    # Track Familiarity
    track_familiarity = add_combined_track_rows(
        original_df=df,
        group_cols=["Track Selection","Track Familiarity"],
        agg_dict={"Email":"count"},
        rename_dict=None,
        compute_rates=lambda d: d.rename(columns={"Email":"Count"}),
        include_unique_participants=False
    )
    track_familiarity = track_familiarity[track_familiarity["Count"]!=0]
    track_familiarity = track_familiarity[~track_familiarity["Track Familiarity"].str.lower().isin(["undefined","unknown","nan"])]
    track_familiarity = track_familiarity.sort_values(["Track Selection","Track Familiarity"]).reset_index(drop=True)

    # CMS Center breakdown
    def compute_cms_center_rates(df_):
        df_["completion_rate"] = df_["Total Completed"]/df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]/df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x:"{:.2%}".format(x))
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x:"{:.2%}".format(x))
        return df_

    cms_center_breakdown = add_combined_track_rows(
        original_df=df,
        group_cols=["CMS Center Selection","Track Selection"],
        agg_dict={"Email":"count","Verified Complete":"sum","Dropped":"sum"},
        rename_dict={
            "Email":"Total Enrolled","Verified Complete":"Total Completed","Dropped":"Total Dropped"
        },
        compute_rates= compute_cms_center_rates,
        include_unique_participants=False
    )
    cms_center_breakdown = cms_center_breakdown.sort_values(["CMS Center Selection","Track Selection"]).reset_index(drop=True)

    # Repeated Enrollments (basic summary)
    repeated_enrollments = df.groupby("Email", observed=False).agg(
        tracks_taken=("Track Selection","nunique"),
        total_enrollments=("Track Selection","count"),
        total_completions=("Verified Complete","sum"),
        total_drops=("Dropped","sum")
    ).reset_index()
    repeated_enrollments = repeated_enrollments[repeated_enrollments["tracks_taken"]>=2]
    multi_track_users = len(repeated_enrollments)

    # Repeated Enrollments By Track
    df_times = df.groupby(["Email","Track Selection"], observed=False).agg(
        times_enrolled=("Track Selection","count"),
        times_completed_track=("Verified Complete","sum")
    ).reset_index()

    repeated_by_email = df.groupby("Email", observed=False).agg(
        total_distinct_tracks=("Track Selection","nunique"),
        total_enrollments=("Track Selection","count"),
        total_completions=("Verified Complete","sum"),
        total_drops=("Dropped","sum")
    ).reset_index()

    repeated_enrollments_by_track = df_times.merge(repeated_by_email, on="Email", how="left").rename(
        columns={
            "Track Selection":"Track",
            "times_enrolled":"Times Enrolled in Track",
            "times_completed_track": "Times Completed in Track"
        }
    )
    repeated_enrollments_by_track = repeated_enrollments_by_track.sort_values(["Track","Email"]).reset_index(drop=True)
    max_repeat_count = repeated_enrollments_by_track["Times Enrolled in Track"].max() if not repeated_enrollments_by_track.empty else 1
    max_track_completions = repeated_enrollments_by_track["Times Completed in Track"].max() if not repeated_enrollments_by_track.empty else 0

    # -------------------------------------------------------------
    # Track + Season + Offering aggregator, plus Overall by Offering
    # -------------------------------------------------------------
    # 1) By Track, Season, Offering aggregator
    # We'll build the aggregator and then filter out "Unknown Offering" + zero enrollments
    df_tso = (
        df.groupby(["Track Selection","Season Selection","Offering"], observed=False)
          .agg(tso_enrolled=("Email","count"),
               tso_completed=("Verified Complete","sum"),
               tso_dropped=("Dropped","sum"))
          .reset_index()
    )
    # Filter out 'Unknown Offering' or 'nan' or 0 enrollments
    # We'll consider "Unknown Offering" if it's literally the string 'Unknown Offering', or blank, or 'nan'
    # (You can adjust as needed if your data has other placeholders).
    df_tso = df_tso[~df_tso["Offering"].str.lower().isin(["unknown offering","nan",""])]
    df_tso = df_tso[df_tso["tso_enrolled"] > 0]

    def compute_tso_rates(row):
        enrolled = row["tso_enrolled"]
        completed = row["tso_completed"]
        dropped = row["tso_dropped"]
        c_rate = (completed / enrolled) if enrolled else 0
        d_rate = (dropped   / enrolled) if enrolled else 0
        return pd.Series({
            "Track Selection": row["Track Selection"],
            "Season Selection": row["Season Selection"],
            "Offering": row["Offering"],
            "Total Enrolled": enrolled,
            "Total Completed": completed,
            "Total Dropped": dropped,
            "completion_rate": f"{c_rate:.2%}",
            "drop_rate": f"{d_rate:.2%}"
        })

    track_season_offering_stats = df_tso.apply(compute_tso_rates, axis=1)
    track_season_offering_stats = track_season_offering_stats.sort_values(
        by=["Track Selection","Season Selection","Offering"]
    ).reset_index(drop=True)
    track_season_offering_records = track_season_offering_stats.to_dict("records")

    # 2) Overall by Offering (all seasons/tracks)
    df_off = (
        df.groupby("Offering", observed=False)
          .agg(
              total_enrolled=("Email","count"),
              total_completed=("Verified Complete","sum"),
              total_dropped=("Dropped","sum")
          )
          .reset_index()
    )
    # Filter out unknown offering or 0 enrollments
    df_off = df_off[~df_off["Offering"].str.lower().isin(["unknown offering","nan",""])]
    df_off = df_off[df_off["total_enrolled"]>0]

    def compute_offering_rates(row):
        e = row["total_enrolled"]
        c = row["total_completed"]
        d = row["total_dropped"]
        cr = c/e if e else 0
        dr = d/e if e else 0
        return pd.Series({
            "Offering": row["Offering"],
            "Total Enrolled": e,
            "Total Completed": c,
            "Total Dropped": d,
            "Completion Rate": f"{cr:.2%}",
            "Drop Rate": f"{dr:.2%}"
        })

    offering_overall = df_off.apply(compute_offering_rates, axis=1).sort_values(by="Offering").reset_index(drop=True)

    # Build HTML for the overall-offering table
    offering_overall_html = []
    offering_overall_html.append("<table id='offeringOverallTable'><thead><tr>")
    for col_ in offering_overall.columns:
        offering_overall_html.append(f"<th>{col_}</th>")
    offering_overall_html.append("</tr></thead><tbody>")
    for _, row_ in offering_overall.iterrows():
        offering_overall_html.append("<tr>")
        for col_ in offering_overall.columns:
            offering_overall_html.append(f"<td>{row_[col_]}</td>")
        offering_overall_html.append("</tr>")
    offering_overall_html.append("</tbody></table>")
    offering_overall_html = "".join(offering_overall_html)

    # Some chart data
    def season_idx_count(df_, col, aggregator='count'):
        grp = df_.groupby("Season Selection", observed=False)[col]
        if aggregator == 'count':
            return grp.count().reindex(season_order, fill_value=0).tolist()
        elif aggregator == 'sum':
            return grp.sum().reindex(season_order, fill_value=0).tolist()
        else:
            raise ValueError("Aggregator must be 'count' or 'sum'.")

    season_enrollment = season_idx_count(df, "Track Selection", aggregator='count')
    season_completions = season_idx_count(df[df["Verified Complete"]==True], "Verified Complete", aggregator='count')
    season_drops = season_idx_count(df[df["Dropped"]==True], "Dropped", aggregator='count')

    # Enrollment Trends per Pillar
    enrollment_trends_pillar_season= df.groupby(["Season Selection","Pillar"], observed=False)["Email"].count().reset_index()
    pivot_pillar= enrollment_trends_pillar_season.pivot(index="Season Selection", columns="Pillar", values="Email").fillna(0)
    pivot_pillar= pivot_pillar.reindex(season_order).fillna(0)
    pillar_datasets=[]
    i=0
    for col_ in pivot_pillar.columns:
        ds= {
            "label": col_,
            "data": pivot_pillar[col_].tolist(),
            "backgroundColor": f"rgba({(i*40)%255}, {(i*80)%255}, {(i*120)%255}, 0.6)",
            "borderColor":     f"rgba({(i*40)%255}, {(i*80)%255}, {(i*120)%255}, 1)",
            "borderWidth":1,
            "fill":False
        }
        i+=1
        pillar_datasets.append(ds)

    # Average Completion Rate per Season
    completion_rate_per_season= df.groupby("Season Selection", observed=False)["Verified Complete"].mean().reset_index()
    completion_rate_per_season= completion_rate_per_season.set_index("Season Selection").reindex(season_order).fillna(0)
    completion_rate_per_season_values= (completion_rate_per_season["Verified Complete"]*100).tolist()

    # Distribution of Tracks across Seasons (counts)
    distribution_tracks_season= df.groupby(["Season Selection","Track Selection"], observed=False)["Email"].count().reset_index()
    pivot_tracks= distribution_tracks_season.pivot(index="Season Selection", columns="Track Selection", values="Email").fillna(0)
    pivot_tracks= pivot_tracks.reindex(season_order).fillna(0)
    distribution_datasets=[]
    i=0
    for col_ in pivot_tracks.columns:
        ds= {
            "label": col_,
            "data": pivot_tracks[col_].tolist(),
            "backgroundColor": f"rgba({(i*30)%255}, {(i*60)%255}, {(i*90)%255}, 0.6)",
            "borderColor":     f"rgba({(i*30)%255}, {(i*60)%255}, {(i*90)%255}, 1)",
            "borderWidth":1,
            "fill":False
        }
        i+=1
        distribution_datasets.append(ds)

    # Build HTML
    html_parts = []

    # HEAD + SIDEBAR + Overall Stats
    html_parts.append(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '    <meta charset="utf-8"/>\n'
        "    <title>Class Participation &amp; Insights Report</title>\n"
        "    <style>\n"
        "        body { margin: 0; font-family: Arial, sans-serif; }\n"
        "        .sidebar {\n"
        "            position: fixed; top: 0; left: 0; width: 220px; height: 100%;\n"
        "            overflow-y: auto; background-color: #f4f4f4; padding: 20px;\n"
        "            border-right: 1px solid #ddd;\n"
        "        }\n"
        "        .sidebar h2 { margin-top: 0; }\n"
        "        .sidebar h3 {\n"
        "            margin-top: 1em;\n"
        "            margin-bottom: 0.5em;\n"
        "            font-size: 1.2em;\n"
        "            border-bottom: 1px solid #ccc;\n"
        "            padding-bottom: 0.3em;\n"
        "        }\n"
        "        .sidebar a {\n"
        "            display: block; margin: 5px 0;\n"
        "            text-decoration: none; color: #333;\n"
        "        }\n"
        "        .content {\n"
        "            margin-left: 240px; padding: 20px;\n"
        "        }\n"
        "        h1 {\n"
        "            font-size: 2em;\n"
        "            margin-top: 1.5em;\n"
        "        }\n"
        "        h2 {\n"
        "            font-size: 1.75em;\n"
        "            margin-top: 1.5em;\n"
        "        }\n"
        "        h3 {\n"
        "            font-size: 1.5em;\n"
        "            margin-top: 1.5em;\n"
        "        }\n"
        "        table {\n"
        "            border-collapse: collapse; margin-bottom: 1.5em;\n"
        "            width: 100%;\n"
        "        }\n"
        "        table, th, td {\n"
        "            border: 1px solid #aaa; padding: 8px;\n"
        "            text-align: left;\n"
        "        }\n"
        "        th {\n"
        "            background-color: #ddd;\n"
        "        }\n"
        "        .filter-container {\n"
        "            margin: 1em 0;\n"
        "        }\n"
        "        .filter-container label {\n"
        "            font-weight: bold; margin-right: 0.5em;\n"
        "        }\n"
        "        .filter-container select {\n"
        "            margin-right: 1em;\n"
        "        }\n"
        "        .section-header {\n"
        "            cursor: pointer;\n"
        "            display: inline-block;\n"
        "            margin-bottom: 0.5em;\n"
        "            padding: 8px 12px;\n"
        "            background-color: #eee;\n"
        "            border: 1px solid #ccc;\n"
        "            border-radius: 4px;\n"
        "            font-weight: bold;\n"
        "            font-size: 1.2em;\n"
        "        }\n"
        "        .section-content {\n"
        "            border: 1px solid #ddd;\n"
        "            padding: 12px;\n"
        "            margin-bottom: 1em;\n"
        "            background-color: #fafafa;\n"
        "        }\n"
        "        .chart-section {\n"
        "            margin-bottom: 2em;\n"
        "        }\n"
        "    </style>\n"
        '    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
        "</head>\n"
        "<body>\n"
        '<div class="sidebar">\n'
        "    <h2>Navigation</h2>\n"
        '    <h3>Tables</h3>\n'
        '    <a href="#overall-stats">Overall Stats</a>\n'
        '    <a href="#overall-followup">Overall Follow-up</a>\n'
        '    <a href="#by-track-section">By Track</a>\n'
        '    <a href="#track-season-offering-section">By Track+Season+Offering</a>\n'
        '    <a href="#offering-overall-section">Overall by Offering</a>\n'
        '    <a href="#combined-track-season-level-stats-section">Combined Track &amp; Season &amp; Level Stats</a>\n'
        '    <a href="#motivations-section">Motivations</a>\n'
        '    <a href="#motivations-by-pillar-section">Motivations by Pillar</a>\n'
        '    <a href="#topic-interest-section">Topic Interest (Full)</a>\n'
        '    <a href="#pillar-section">Pillar Insights</a>\n'
        '    <a href="#special-section">Special Interests</a>\n'
        '    <a href="#ba-familiarity-section">BA Familiarity</a>\n'
        '    <a href="#benefits-agile-section">Benefits of Agile</a>\n'
        '    <a href="#apps-agile-section">Applications of Agile</a>\n'
        '    <a href="#track-familiarity-section">Track Familiarity</a>\n'
        '    <a href="#cms-center-section">CMS Center Breakdown</a>\n'
        '    <a href="#repeated-enrollments-section">Repeated Enrollments</a>\n'
        '    <a href="#repeated-enrollments-by-track-section">Repeated Enrollments (By Track)</a>\n'
        '    <h3>Charts</h3>\n'
        '    <a href="#charts-section">Charts Overview</a>\n'
        '    <a href="#season-enrollment-trend-section">Seasonal Enrollment Trends</a>\n'
        '    <a href="#enrollment-trends-pillar-season-section">Enrollment Trends (Pillar)</a>\n'
        '    <a href="#average-completion-rate-section">Average Completion Rate</a>\n'
        '    <a href="#distribution-tracks-season-section">Tracks across Seasons</a>\n'
        "</div>\n"
        '<div class="content">\n'
        '    <h1 id="overall-stats">Overall Stats</h1>\n'
        f"    <p><strong>Total Registrations:</strong> {total_registrations}</p>\n"
        f"    <p><strong>Total Unique Participants:</strong> {total_unique_participants}</p>\n"
        f"    <p><strong>Total Dropped:</strong> {total_dropped} (Rate: {drop_rate:.2%})</p>\n"
        f"    <p><strong>Total Completed:</strong> {total_completed} (Rate: {completion_rate:.2%})</p>\n"
        '    <hr/><h1 id="overall-followup">Overall Follow-up</h1>\n'
        f"    <p>Participants who completed ≥1 track: {num_completed}</p>\n"
        f"    <p>Those who enrolled in more than one track: {num_followed_up}</p>\n"
        f"    <p><strong>Overall Follow-up Rate:</strong> {overall_follow_up_rate:.2%}</p>\n"
    )

    # ----- “By Track” table -----
    html_parts.append(
        "<hr/><div id=\"by-track-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('byTrackContent')\">By Track (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"byTrackContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"byTrackFilter\">Filter by Track:</label>\n"
        "    <select id=\"byTrackFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    track_list_for_filter = sorted(track_stats["Track Selection"].unique())
    for t_ in track_list_for_filter:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"byTrackTable\">\n"
        "<thead><tr>\n"
    )
    for col in track_stats.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in track_stats.iterrows():
        track_sel = row["Track Selection"]
        html_parts.append(f"<tr data-track=\"{track_sel}\">")
        for col in track_stats.columns:
            val = row[col]
            html_parts.append(f"<td>{val}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # NEW SECTION: By Track+Season+Offering
    html_parts.append(
        "<hr/><div id=\"track-season-offering-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('trackSeasonOfferingContent')\">"
        "By Track, Season, Offering (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"trackSeasonOfferingContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"tsoTrackFilter\">Select Track(s) (Ctrl+Click):</label>\n"
        "    <select id=\"tsoTrackFilter\" multiple size=\"5\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    tso_unique_tracks = sorted(track_season_offering_stats["Track Selection"].unique())
    for t_ in tso_unique_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "    <label for=\"tsoSeasonFilter\">Select Seasons (Ctrl+Click):</label>\n"
        "    <select id=\"tsoSeasonFilter\" multiple size=\"5\">\n"
    )
    # We keep the known seasons in the multi-select
    for s_ in season_order:
        html_parts.append(f"        <option value=\"{s_}\">{s_}</option>\n")

    # Unique offerings (already filtered from aggregator, so "Unknown" won't appear here)
    tso_unique_offerings = sorted(track_season_offering_stats["Offering"].unique())
    html_parts.append(
        "    </select>\n"
        "    <label for=\"tsoOfferingFilter\">Select Offering(s) (Ctrl+Click):</label>\n"
        "    <select id=\"tsoOfferingFilter\" multiple size=\"5\">\n"
        "        <option value=\"ALL\">All Offerings</option>\n"
    )
    for off_ in tso_unique_offerings:
        html_parts.append(f"        <option value=\"{off_}\">{off_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "    <button onclick=\"updateTrackSeasonOfferingStats()\">Update</button>\n"
        "</div>\n"
        "<div id=\"trackSeasonOfferingResults\"></div>\n"
        "</div></div>\n"
    )

    # NEW SECTION: Overall by Offering
    html_parts.append(
        "<hr/><div id=\"offering-overall-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('offeringOverallContent')\">"
        "Overall by Offering (All Seasons) (Click to Minimize/Expand)</div>\n"
        f"<div class=\"section-content\" id=\"offeringOverallContent\">\n{offering_overall_html}\n</div></div>\n"
    )

    # Combined aggregator (Season + Track + Level)
    html_parts.append(
        "<hr/><div id=\"combined-track-season-level-stats-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('combinedTrackSeasonLevelContent')\">"
        "Combined Track &amp; Season &amp; Level Stats (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"combinedTrackSeasonLevelContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"ctslTrackFilter\">Select Tracks (Ctrl+Click):</label>\n"
        "    <select id=\"ctslTrackFilter\" multiple size=\"5\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    st_unique_tracks = sorted(season_track_level_stats["Track Selection"].unique())
    for t_ in st_unique_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "    <label for=\"ctslSeasonFilter\">Select Seasons (Ctrl+Click):</label>\n"
        "    <select id=\"ctslSeasonFilter\" multiple size=\"5\">\n"
    )
    for s_ in season_order:
        html_parts.append(f"        <option value=\"{s_}\">{s_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "    <label for=\"ctslLevelFilter\">Select Levels (Ctrl+Click):</label>\n"
        "    <select id=\"ctslLevelFilter\" multiple size=\"5\">\n"
        "        <option value=\"ALL\">All Levels</option>\n"
    )
    all_levels_sorted = sorted(season_track_level_stats["Level"].unique())
    for lv_ in all_levels_sorted:
        html_parts.append(f"        <option value=\"{lv_}\">{lv_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "    <button onclick=\"updateCombinedTSLStats()\">Update</button>\n"
        "</div>\n"
        "<div id=\"combinedTSLResults\"></div>\n"
        "</div></div>\n"
    )

    # Motivations (top 5 per track)
    html_parts.append(
        "<hr/><div id=\"motivations-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('motivationsContent')\">"
        "Motivations (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"motivationsContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"motivationsFilter\">Filter by Track:</label>\n"
        "    <select id=\"motivationsFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    mot_tracks = sorted(motivation_by_track["Track Selection"].unique())
    for t_ in mot_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"motivationsTable\"><thead><tr>\n"
    )
    for col in motivation_by_track.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in motivation_by_track.iterrows():
        trk_ = row["Track Selection"]
        mentions_ = row["Mentions"]
        if mentions_ == 0:
            continue
        html_parts.append(f"<tr data-track=\"{trk_}\">")
        for col in motivation_by_track.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # Motivations by Pillar (Top 5)
    html_parts.append(
        "<hr/><div id=\"motivations-by-pillar-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('motivationsByPillarContent')\">"
        "Motivations by Pillar (Top 5)</div>\n"
        "<div class=\"section-content\" id=\"motivationsByPillarContent\">\n"
        "<table><thead><tr><th>Pillar</th><th>Motivation</th><th>Count</th></tr></thead><tbody>\n"
    )
    for _, row in motivations_trends_by_pillar.iterrows():
        if row["Count"] == 0:
            continue
        html_parts.append(
            f"<tr><td>{row['Pillar']}</td>"
            f"<td>{row['Motivation']}</td>"
            f"<td>{row['Count']}</td></tr>\n"
        )
    html_parts.append("</tbody></table>\n</div></div>\n")

    # Topic Interest
    html_parts.append(
        "<hr/><div id=\"topic-interest-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('topicInterestContent')\">"
        "Topic Interest (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"topicInterestContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"topicInterestFilter\">Filter by Track:</label>\n"
        "    <select id=\"topicInterestFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    ti_tracks = sorted(topic_interest["Track Selection"].unique())
    for t_ in ti_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"topicInterestTable\"><thead><tr>\n"
    )
    for col in topic_interest.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in topic_interest.iterrows():
        trk_ = row["Track Selection"]
        cnt_ = row["Count"]
        if cnt_ == 0:
            continue
        html_parts.append(f"<tr data-track=\"{trk_}\">")
        for col in topic_interest.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # Pillar Insights
    html_parts.append(
        "<hr/><div id=\"pillar-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('pillarContent')\">"
        "Pillar Insights (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"pillarContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"pillarFilter\">Filter by Track:</label>\n"
        "    <select id=\"pillarFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    unique_pillar_tracks = sorted(pillar_by_track_agg["Track Selection"].unique())
    for t_ in unique_pillar_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"pillarTable\"><thead><tr>\n"
    )
    for col in pillar_by_track_agg.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in pillar_by_track_agg.iterrows():
        trk_ = row["Track Selection"]
        count_ = row["Count"]
        if count_ == 0:
            continue
        html_parts.append(f"<tr data-track=\"{trk_}\">")
        for col in pillar_by_track_agg.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # Special Interests
    special_by_track = add_combined_track_rows(
        original_df=df,
        group_cols=["Track Selection","Special Interests"],
        agg_dict={"Email":"count"},
        rename_dict=None,
        compute_rates=lambda d: d.rename(columns={"Email":"Count"}),
        include_unique_participants=False
    )
    special_by_track = special_by_track[special_by_track["Count"]!=0]
    special_by_track = special_by_track[~special_by_track["Special Interests"].str.lower().isin(["undefined","unknown","nan"])]
    special_by_track = special_by_track.sort_values(["Track Selection","Special Interests"]).reset_index(drop=True)

    html_parts.append(
        "<hr/><div id=\"special-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('specialContent')\">"
        "Special Interests (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"specialContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"specialFilter\">Filter by Track:</label>\n"
        "    <select id=\"specialFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    unique_spec_tracks = sorted(special_by_track["Track Selection"].unique())
    for t_ in unique_spec_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"specialTable\"><thead><tr>\n"
    )
    for col in special_by_track.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in special_by_track.iterrows():
        trk_ = row["Track Selection"]
        cnt_ = row["Count"]
        if cnt_ == 0:
            continue
        html_parts.append(f"<tr data-track=\"{trk_}\">")
        for col in special_by_track.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # BA Familiarity, Benefits, Applications
    html_parts.append(
        "<hr/><div id=\"ba-familiarity-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('baFamiliarityContent')\">"
        "BA Familiarity (Business Agility) (Click to Minimize/Expand)</div>\n"
        f"<div class=\"section-content\" id=\"baFamiliarityContent\">\n{ba_familiarity_by_track.to_html(index=False)}\n</div></div>\n"
        "<hr/><div id=\"benefits-agile-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('benefitsAgileContent')\">"
        "Benefits of Agile (Business Agility) (Click to Minimize/Expand)</div>\n"
        f"<div class=\"section-content\" id=\"benefitsAgileContent\">\n{benefits_agile_by_track.to_html(index=False)}\n</div></div>\n"
        "<hr/><div id=\"apps-agile-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('appsAgileContent')\">"
        "Applications of Agile (Business Agility) (Click to Minimize/Expand)</div>\n"
        f"<div class=\"section-content\" id=\"appsAgileContent\">\n{apps_agile_by_track.to_html(index=False)}\n</div></div>\n"
    )

    # Track Familiarity
    html_parts.append(
        "<hr/><div id=\"track-familiarity-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('trackFamiliarityContent')\">"
        "Track Familiarity (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"trackFamiliarityContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"trackFamiliarityFilter\">Filter by Track:</label>\n"
        "    <select id=\"trackFamiliarityFilter\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    unique_fam_tracks = sorted(track_familiarity["Track Selection"].unique())
    for t_ in unique_fam_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"trackFamiliarityTable\"><thead><tr>\n"
    )
    for col in track_familiarity.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in track_familiarity.iterrows():
        trk_ = row["Track Selection"]
        if row["Count"] == 0:
            continue
        html_parts.append(f"<tr data-track=\"{trk_}\">")
        for col in track_familiarity.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # CMS Center breakdown
    html_parts.append(
        "<hr/><div id=\"cms-center-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('cmsCenterContent')\">"
        "CMS Center Breakdown (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"cmsCenterContent\">\n"
        "<div class=\"filter-container\">\n"
        "    <label for=\"cmsCenterFilterTrack\">Filter by Track:</label>\n"
        "    <select id=\"cmsCenterFilterTrack\">\n"
        "        <option value=\"ALL\">All Tracks</option>\n"
    )
    unique_cms_tracks = sorted(cms_center_breakdown["Track Selection"].unique())
    for t_ in unique_cms_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    unique_centers = sorted(cms_center_breakdown["CMS Center Selection"].unique())
    html_parts.append(
        "    </select>\n\n"
        "    <label for=\"cmsCenterFilterCenter\">Filter by CMS Center:</label>\n"
        "    <select id=\"cmsCenterFilterCenter\">\n"
        "        <option value=\"ALL\">All Centers</option>\n"
    )
    for c_ in unique_centers:
        html_parts.append(f"        <option value=\"{c_}\">{c_}</option>\n")

    html_parts.append(
        "    </select>\n"
        "</div>\n"
        "<table id=\"cmsCenterTable\"><thead><tr>\n"
    )
    for col in cms_center_breakdown.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in cms_center_breakdown.iterrows():
        trk_ = row["Track Selection"]
        cent_ = row["CMS Center Selection"]
        html_parts.append(f"<tr data-track=\"{trk_}\" data-center=\"{cent_}\">")
        for col in cms_center_breakdown.columns:
            html_parts.append(f"<td>{row[col]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # Repeated Enrollments (basic)
    html_parts.append(
        "<hr/><div id=\"repeated-enrollments-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('repeatedEnrollmentsContent')\">"
        "Repeated Enrollments (Click to Minimize/Expand)</div>\n"
        f"<div class=\"section-content\" id=\"repeatedEnrollmentsContent\">\n"
        f"<p>Number of participants who enrolled in multiple distinct tracks: {multi_track_users}</p>\n"
        f"{repeated_enrollments.to_html(index=False)}\n"
        "</div></div>\n"
    )

    # Repeated Enrollments By Track
    html_parts.append(
        "<hr/><div id=\"repeated-enrollments-by-track-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('repeatedByTrackContent')\">"
        "Repeated Enrollments (By Track) (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"repeatedByTrackContent\">\n"
        "<div class=\"filter-container\" style=\"margin-bottom:1em;\">\n"
        "    <label for=\"repeatedByTrackFilter\">Select Track(s) (Ctrl+Click):</label>\n"
        "    <select id=\"repeatedByTrackFilter\" multiple size=\"5\">\n"
        '        <option value="ALL">All Tracks</option>\n'
    )
    rep_tracks = sorted(repeated_enrollments_by_track["Track"].unique())
    for t_ in rep_tracks:
        html_parts.append(f"        <option value=\"{t_}\">{t_}</option>\n")

    html_parts.append(
        "    </select>\n\n"
        "    <label for=\"repeatsCountFilter\">Select Repeat Count(s) (Ctrl+Click):</label>\n"
        "    <select id=\"repeatsCountFilter\" multiple size=\"5\">\n"
    )
    for c_ in range(1, max_repeat_count+1):
        html_parts.append(f"        <option value=\"{c_}\">{c_}</option>\n")

    html_parts.append(
        "    </select>\n\n"
        "    <label for=\"completionsCountFilter\">Select Completion Count(s) (Ctrl+Click):</label>\n"
        "    <select id=\"completionsCountFilter\" multiple size=\"5\">\n"
    )
    for c_ in range(0, max_track_completions+1):
        html_parts.append(f"        <option value=\"{c_}\">{c_}</option>\n")

    html_parts.append(
        "    </select>\n\n"
        "    <button onclick=\"updateRepeatedByTrackStats()\">Update</button>\n"
        "</div>\n"
        "<div id=\"repeatedByTrackResults\"></div>\n"
        "</div></div>\n"
    )

    # Charts section
    html_parts.append(
        "<hr/><div id=\"charts-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('chartsContent')\">Charts (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"chartsContent\">\n"

        "<div class=\"chart-section\" id=\"season-enrollment-trend-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('seasonEnrollmentTrendContent')\">"
        "Seasonal Enrollment Trends (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"seasonEnrollmentTrendContent\">\n"
        "<canvas id=\"seasonEnrollmentChart\" width=\"800\" height=\"400\"></canvas>\n"
        "</div></div>\n"

        "<div class=\"chart-section\" id=\"enrollment-trends-pillar-season-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('enrollmentTrendsPillarSeasonContent')\">"
        "Enrollment Trends per Pillar over Seasons (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"enrollmentTrendsPillarSeasonContent\">\n"
        "<canvas id=\"enrollmentTrendsPillarSeasonChart\" width=\"800\" height=\"400\"></canvas>\n"
        "</div></div>\n"

        "<div class=\"chart-section\" id=\"average-completion-rate-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('averageCompletionRateContent')\">"
        "Average Completion Rate per Season (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"averageCompletionRateContent\">\n"
        "<canvas id=\"averageCompletionRateChart\" width=\"800\" height=\"400\"></canvas>\n"
        "</div></div>\n"

        "<div class=\"chart-section\" id=\"distribution-tracks-season-section\">\n"
        "<div class=\"section-header\" onclick=\"toggleSection('distributionTracksSeasonContent')\">"
        "Distribution of Tracks across Seasons (Click to Minimize/Expand)</div>\n"
        "<div class=\"section-content\" id=\"distributionTracksSeasonContent\">\n"
        "<canvas id=\"distributionTracksSeasonChart\" width=\"800\" height=\"400\"></canvas>\n"
        "</div></div>\n"

        "</div></div>\n"
    )

    # Data for JS
    season_order_json            = json.dumps(season_order)
    season_enrollment_json       = json.dumps(season_enrollment)
    season_completions_json      = json.dumps(season_completions)
    season_drops_json            = json.dumps(season_drops)
    pillar_datasets_json         = json.dumps(pillar_datasets)
    completion_rate_values_json  = json.dumps(completion_rate_per_season_values)
    distribution_datasets_json   = json.dumps(distribution_datasets)

    season_track_level_records   = season_track_level_stats.to_dict("records")
    season_track_level_json      = json.dumps(season_track_level_records)

    repeated_by_track_records    = repeated_enrollments_by_track.to_dict("records")
    repeated_by_track_json       = json.dumps(repeated_by_track_records)

    tso_data_json = json.dumps(track_season_offering_records)

    # JS script
    html_parts.append(
        "<script>\n"
        "function toggleSection(contentId) {\n"
        "    const contentDiv = document.getElementById(contentId);\n"
        "    contentDiv.style.display = (contentDiv.style.display === 'none') ? 'block' : 'none';\n"
        "}\n\n"
        "// By Track filter\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "    const trackSelect = document.getElementById('byTrackFilter');\n"
        "    const table = document.getElementById('byTrackTable');\n"
        "    const rows = table.querySelectorAll('tbody tr');\n\n"
        "    trackSelect.addEventListener('change', function() {\n"
        "        const chosenTrack = trackSelect.value;\n"
        "        rows.forEach(function(r) {\n"
        "            const rowTrack = r.getAttribute('data-track');\n"
        "            r.style.display = (chosenTrack==='ALL' || rowTrack===chosenTrack) ? '' : 'none';\n"
        "        });\n"
        "    });\n"
        "});\n\n"
        "// Generic function to add a track dropdown filter\n"
        "function addTrackFilterListener(tableId, dropdownId) {\n"
        "    const sel = document.getElementById(dropdownId);\n"
        "    sel.addEventListener('change', function() {\n"
        "        const chosenTrack = this.value;\n"
        "        const table = document.getElementById(tableId);\n"
        "        const rows = table.querySelectorAll('tbody tr');\n"
        "        rows.forEach(function(r) {\n"
        "            const rowTrack = r.getAttribute('data-track');\n"
        "            r.style.display = (chosenTrack==='ALL' || rowTrack===chosenTrack) ? '' : 'none';\n"
        "        });\n"
        "    });\n"
        "}\n"
        "// Add track-based filters for different tables\n"
        "addTrackFilterListener('motivationsTable','motivationsFilter');\n"
        "addTrackFilterListener('topicInterestTable','topicInterestFilter');\n"
        "addTrackFilterListener('pillarTable','pillarFilter');\n"
        "addTrackFilterListener('specialTable','specialFilter');\n"
        "addTrackFilterListener('trackFamiliarityTable','trackFamiliarityFilter');\n\n"
        "// CMS Center table => Filter by (Track, Center)\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "    const trackSelect = document.getElementById('cmsCenterFilterTrack');\n"
        "    const centerSelect= document.getElementById('cmsCenterFilterCenter');\n"
        "    const table = document.getElementById('cmsCenterTable');\n"
        "    const rows = table.querySelectorAll('tbody tr');\n\n"
        "    function filterCMSCenterTable() {\n"
        "        const chosenTrack = trackSelect.value;\n"
        "        const chosenCenter= centerSelect.value;\n"
        "        rows.forEach(function(r) {\n"
        "            const rowTrack = r.getAttribute('data-track');\n"
        "            const rowCenter= r.getAttribute('data-center');\n"
        "            const matchTrack = (chosenTrack==='ALL' || rowTrack===chosenTrack);\n"
        "            const matchCenter= (chosenCenter==='ALL' || rowCenter===chosenCenter);\n"
        "            r.style.display= (matchTrack && matchCenter)? '' : 'none';\n"
        "        });\n"
        "    }\n\n"
        "    trackSelect.addEventListener('change', filterCMSCenterTable);\n"
        "    centerSelect.addEventListener('change', filterCMSCenterTable);\n"
        "});\n\n"
        "// CHARTS\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "    var ctxSeason = document.getElementById('seasonEnrollmentChart').getContext('2d');\n"
        "    new Chart(ctxSeason, {\n"
        "        type: 'line',\n"
        "        data: {\n"
        f"            labels: {season_order_json},\n"
        "            datasets: [\n"
        "               {\n"
        "                   label: 'Total Enrolled',\n"
        f"                   data: {season_enrollment_json},\n"
        "                   backgroundColor: 'rgba(54,162,235,0.2)',\n"
        "                   borderColor: 'rgba(54,162,235,1)',\n"
        "                   borderWidth:1,\n"
        "                   fill:false\n"
        "               },\n"
        "               {\n"
        "                   label: 'Total Completed',\n"
        f"                   data: {season_completions_json},\n"
        "                   backgroundColor: 'rgba(75,192,192,0.2)',\n"
        "                   borderColor: 'rgba(75,192,192,1)',\n"
        "                   borderWidth:1,\n"
        "                   fill:false\n"
        "               },\n"
        "               {\n"
        "                   label: 'Total Dropped',\n"
        f"                   data: {season_drops_json},\n"
        "                   backgroundColor: 'rgba(255,99,132,0.2)',\n"
        "                   borderColor: 'rgba(255,99,132,1)',\n"
        "                   borderWidth:1,\n"
        "                   fill:false\n"
        "               }\n"
        "            ]\n"
        "        },\n"
        "        options: {\n"
        "            responsive:true,\n"
        "            plugins: {\n"
        "                title: {\n"
        "                    display:true,\n"
        "                    text:'Seasonal Enrollment Trends'\n"
        "                }\n"
        "            },\n"
        "            scales: {\n"
        "                y: {\n"
        "                    beginAtZero:true,\n"
        "                    ticks: {\n"
        "                        precision:0\n"
        "                    }\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    });\n\n"
        "    // Enrollment Trends per Pillar\n"
        "    var ctxEnrollmentPillar = document.getElementById('enrollmentTrendsPillarSeasonChart').getContext('2d');\n"
        "    new Chart(ctxEnrollmentPillar, {\n"
        "        type:'bar',\n"
        "        data: {\n"
        f"            labels: {season_order_json},\n"
        f"            datasets: {pillar_datasets_json}\n"
        "        },\n"
        "        options: {\n"
        "            responsive:true,\n"
        "            plugins: {\n"
        "                title: {\n"
        "                    display:true,\n"
        "                    text:'Enrollment Trends per Pillar over Seasons'\n"
        "                },\n"
        "                tooltip: {\n"
        "                    mode:'index',\n"
        "                    intersect:false\n"
        "                }\n"
        "            },\n"
        "            scales: {\n"
        "                x: {\n"
        "                    stacked:true\n"
        "                },\n"
        "                y: {\n"
        "                    stacked:true,\n"
        "                    beginAtZero:true\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    });\n\n"
        "    // Average Completion Rate per Season\n"
        "    var ctxAvgCompletion = document.getElementById('averageCompletionRateChart').getContext('2d');\n"
        "    new Chart(ctxAvgCompletion, {\n"
        "        type:'line',\n"
        "        data: {\n"
        f"            labels: {season_order_json},\n"
        "            datasets:[{\n"
        "                label:'Average Completion Rate (%)',\n"
        f"                data:{completion_rate_values_json},\n"
        "                backgroundColor:'rgba(153,102,255,0.2)',\n"
        "                borderColor:'rgba(153,102,255,1)',\n"
        "                borderWidth:1,\n"
        "                fill:false,\n"
        "                tension:0.1\n"
        "            }]\n"
        "        },\n"
        "        options: {\n"
        "            responsive:true,\n"
        "            plugins: {\n"
        "                title: {\n"
        "                    display:true,\n"
        "                    text:'Average Completion Rate per Season'\n"
        "                },\n"
        "                tooltip: {\n"
        "                    callbacks: {\n"
        "                        label:function(ctx) { return ctx.parsed.y + '%'; }\n"
        "                    }\n"
        "                }\n"
        "            },\n"
        "            scales: {\n"
        "                y: {\n"
        "                    beginAtZero:true,\n"
        "                    max:100,\n"
        "                    ticks: {\n"
        "                        callback:function(value) { return value + '%'; }\n"
        "                    }\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    });\n\n"
        "    // Distribution of Tracks across Seasons\n"
        "    var ctxDist = document.getElementById('distributionTracksSeasonChart').getContext('2d');\n"
        "    new Chart(ctxDist, {\n"
        "        type:'bar',\n"
        "        data: {\n"
        f"            labels: {season_order_json},\n"
        f"            datasets: {distribution_datasets_json}\n"
        "        },\n"
        "        options: {\n"
        "            responsive:true,\n"
        "            plugins: {\n"
        "                title: {\n"
        "                    display:true,\n"
        "                    text:'Distribution of Tracks across Seasons'\n"
        "                },\n"
        "                tooltip: {\n"
        "                    mode:'index',\n"
        "                    intersect:false\n"
        "                }\n"
        "            },\n"
        "            scales: {\n"
        "                x: {\n"
        "                    stacked:false\n"
        "                },\n"
        "                y: {\n"
        "                    beginAtZero:true\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    });\n"
        "});\n\n"
        "// Combined aggregator for (Track, Season, Level)\n"
        f"var seasonOrder = {season_order_json};\n"
        f"var seasonTrackLevelData = {season_track_level_json};\n"
        "function updateCombinedTSLStats() {\n"
        "    var trackSelect = document.getElementById('ctslTrackFilter');\n"
        "    var selectedTracks = Array.from(trackSelect.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedTracks.includes('ALL') || selectedTracks.length===0) {\n"
        "        selectedTracks=null;\n"
        "    }\n"
        "    var seasonSelect = document.getElementById('ctslSeasonFilter');\n"
        "    var selectedSeasons = Array.from(seasonSelect.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedSeasons.length===0) {\n"
        "        selectedSeasons=seasonOrder;\n"
        "    }\n"
        "    var levelSelect = document.getElementById('ctslLevelFilter');\n"
        "    var selectedLevels = Array.from(levelSelect.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedLevels.includes('ALL') || selectedLevels.length===0) {\n"
        "        selectedLevels=null;\n"
        "    }\n\n"
        "    var filtered = seasonTrackLevelData.filter(function(row) {\n"
        "        var trackMatch=true;\n"
        "        if(selectedTracks) {\n"
        "            trackMatch = selectedTracks.includes(row['Track Selection']);\n"
        "        }\n"
        "        var seasonMatch = selectedSeasons.includes(row['Season Selection']);\n"
        "        var levelMatch=true;\n"
        "        if(selectedLevels) {\n"
        "            levelMatch = selectedLevels.includes(row['Level']);\n"
        "        }\n"
        "        return trackMatch && seasonMatch && levelMatch;\n"
        "    });\n\n"
        "    var nonZero = filtered.filter(function(r){\n"
        "       return parseFloat(r['Total Enrolled']) > 0;\n"
        "    });\n"
        "    var html=\"\";\n"
        "    if(nonZero.length===0) {\n"
        "        html=\"<p>No data found for these filters.</p>\";\n"
        "    } else {\n"
        "        html += \"<h3>Individual Stats</h3>\";\n"
        "        html += \"<table><thead><tr><th>Season</th><th>Track</th><th>Level</th>"
        "<th>Total Enrolled</th><th>Total Completed</th><th>Total Dropped</th>"
        "<th>Completion Rate</th><th>Drop Rate</th><th>unique_participants</th></tr></thead><tbody>\";\n"
        "        let sumEnrolled=0, sumCompleted=0, sumDropped=0;\n"
        "        nonZero.forEach(function(r) {\n"
        "            html += \"<tr>\";\n"
        "            html += \"<td>\"+r['Season Selection']+\"</td>\";\n"
        "            html += \"<td>\"+r['Track Selection']+\"</td>\";\n"
        "            html += \"<td>\"+r['Level']+\"</td>\";\n"
        "            html += \"<td>\"+r['Total Enrolled']+\"</td>\";\n"
        "            html += \"<td>\"+r['Total Completed']+\"</td>\";\n"
        "            html += \"<td>\"+r['Total Dropped']+\"</td>\";\n"
        "            html += \"<td>\"+(r['completion_rate'] || '0.00%')+\"</td>\";\n"
        "            html += \"<td>\"+(r['drop_rate'] || '0.00%')+\"</td>\";\n"
        "            html += \"<td>\"+(r['unique_participants'] || '')+\"</td>\";\n"
        "            html += \"</tr>\";\n"
        "            let enr = parseFloat(r['Total Enrolled'])||0;\n"
        "            let cmp = parseFloat(r['Total Completed'])||0;\n"
        "            let drp = parseFloat(r['Total Dropped'])||0;\n"
        "            sumEnrolled += enr;\n"
        "            sumCompleted += cmp;\n"
        "            sumDropped   += drp;\n"
        "        });\n"
        "        html += \"</tbody></table>\";\n\n"
        "        let cRate = sumEnrolled ? (sumCompleted/sumEnrolled) : 0;\n"
        "        let dRate = sumEnrolled ? (sumDropped/sumEnrolled) : 0;\n"
        "        html += \"<h3>Combined Stats</h3>\";\n"
        "        html += \"<p>Total Enrolled: \"+sumEnrolled+\"</p>\";\n"
        "        html += \"<p>Total Completed: \"+sumCompleted+\"</p>\";\n"
        "        html += \"<p>Total Dropped: \"+sumDropped+\"</p>\";\n"
        "        html += \"<p>Combined Completion Rate: \"+(cRate*100).toFixed(2)+\"%</p>\";\n"
        "        html += \"<p>Combined Drop Rate: \"+(dRate*100).toFixed(2)+\"%</p>\";\n"
        "    }\n"
        "    document.getElementById('combinedTSLResults').innerHTML= html;\n"
        "}\n\n"
        "// Repeated Enrollments By Track\n"
        f"var repeatedByTrackData = {repeated_by_track_json};\n"
        "function updateRepeatedByTrackStats() {\n"
        "    var trackSel = document.getElementById('repeatedByTrackFilter');\n"
        "    var selectedTracks = Array.from(trackSel.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedTracks.includes('ALL') || selectedTracks.length===0) {\n"
        "        selectedTracks=null;\n"
        "    }\n"
        "    var repeatsSel = document.getElementById('repeatsCountFilter');\n"
        "    var selectedRepeats = Array.from(repeatsSel.selectedOptions).map(opt => parseInt(opt.value));\n"
        "    if(selectedRepeats.length===0) {\n"
        "        selectedRepeats=null;\n"
        "    }\n"
        "    var completionsSel = document.getElementById('completionsCountFilter');\n"
        "    var selectedCompletions = Array.from(completionsSel.selectedOptions).map(opt => parseInt(opt.value));\n"
        "    if(selectedCompletions.length===0) {\n"
        "        selectedCompletions=null;\n"
        "    }\n\n"
        "    var filtered = repeatedByTrackData;\n"
        "    if(selectedTracks) {\n"
        "        filtered = filtered.filter(function(r){\n"
        "           return selectedTracks.includes(r['Track']);\n"
        "        });\n"
        "    }\n"
        "    if(selectedRepeats) {\n"
        "        filtered = filtered.filter(function(r){\n"
        "            return selectedRepeats.includes(r['Times Enrolled in Track']);\n"
        "        });\n"
        "    }\n"
        "    if(selectedCompletions) {\n"
        "        filtered = filtered.filter(function(r){\n"
        "            return selectedCompletions.includes(r['Times Completed in Track']);\n"
        "        });\n"
        "    }\n"
        "    if(filtered.length===0) {\n"
        "        document.getElementById('repeatedByTrackResults').innerHTML = '<p>No data found for these filters.</p>';\n"
        "        return;\n"
        "    }\n"
        "    var uniqueEmails = new Set(filtered.map(x => x['Email']));\n"
        "    var countUnique = uniqueEmails.size;\n\n"
        "    var html = `<p>Number of participants matching these filters: <strong>${countUnique}</strong></p>`;\n"
        "    html += '<table><thead><tr>';\n"
        "    var columns = [\n"
        "       'Email','Track','Times Enrolled in Track','Times Completed in Track',\n"
        "       'total_distinct_tracks','total_enrollments','total_completions','total_drops'\n"
        "    ];\n"
        "    columns.forEach(col => {\n"
        "        html += `<th>${col}</th>`;\n"
        "    });\n"
        "    html += '</tr></thead><tbody>';\n"
        "    filtered.forEach(row => {\n"
        "        html += '<tr>';\n"
        "        columns.forEach(col => {\n"
        "            html += `<td>${row[col]}</td>`;\n"
        "        });\n"
        "        html += '</tr>';\n"
        "    });\n"
        "    html += '</tbody></table>';\n"
        "    document.getElementById('repeatedByTrackResults').innerHTML = html;\n"
        "}\n\n"
        "// Track-Season-Offering aggregator\n"
        f"var trackSeasonOfferingData = {tso_data_json};\n"
        "function updateTrackSeasonOfferingStats() {\n"
        "    var trackSel = document.getElementById('tsoTrackFilter');\n"
        "    var selectedTracks = Array.from(trackSel.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedTracks.includes('ALL') || selectedTracks.length===0) {\n"
        "        selectedTracks=null;\n"
        "    }\n"
        "    var seasonSel = document.getElementById('tsoSeasonFilter');\n"
        "    var selectedSeasons = Array.from(seasonSel.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedSeasons.length===0) {\n"
        "        // if none chosen, interpret as all\n"
        "        selectedSeasons = trackSeasonOfferingData.map(x => x['Season Selection']);\n"
        "    }\n"
        "    var offeringSel = document.getElementById('tsoOfferingFilter');\n"
        "    var selectedOfferings = Array.from(offeringSel.selectedOptions).map(opt => opt.value);\n"
        "    if(selectedOfferings.includes('ALL') || selectedOfferings.length===0) {\n"
        "        selectedOfferings=null;\n"
        "    }\n\n"
        "    var filtered = trackSeasonOfferingData.filter(function(row){\n"
        "        let matchTrack = true;\n"
        "        if(selectedTracks) {\n"
        "            matchTrack = selectedTracks.includes(row['Track Selection']);\n"
        "        }\n"
        "        let matchSeason = selectedSeasons.includes(row['Season Selection']);\n"
        "        let matchOffering = true;\n"
        "        if(selectedOfferings) {\n"
        "            matchOffering = selectedOfferings.includes(row['Offering']);\n"
        "        }\n"
        "        return matchTrack && matchSeason && matchOffering;\n"
        "    });\n\n"
        "    if(filtered.length===0) {\n"
        "        document.getElementById('trackSeasonOfferingResults').innerHTML = '<p>No data found for these filters.</p>';\n"
        "        return;\n"
        "    }\n\n"
        "    var html = '<table><thead><tr>';\n"
        "    var cols = [\n"
        "       'Track Selection','Season Selection','Offering','Total Enrolled',\n"
        "       'Total Completed','Total Dropped','completion_rate','drop_rate'\n"
        "    ];\n"
        "    cols.forEach(c => { html += `<th>${c}</th>`; });\n"
        "    html += '</tr></thead><tbody>';\n"
        "    let sumEnrolled=0, sumCompleted=0, sumDropped=0;\n"
        "    filtered.forEach(row => {\n"
        "       html += '<tr>';\n"
        "       cols.forEach(c => {\n"
        "          html += `<td>${row[c]}</td>`;\n"
        "       });\n"
        "       html += '</tr>';\n"
        "       let e = parseFloat(row['Total Enrolled'])||0;\n"
        "       let comp = parseFloat(row['Total Completed'])||0;\n"
        "       let drp = parseFloat(row['Total Dropped'])||0;\n"
        "       sumEnrolled += e;\n"
        "       sumCompleted += comp;\n"
        "       sumDropped += drp;\n"
        "    });\n"
        "    html += '</tbody></table>';\n\n"
        "    let cRate = sumEnrolled ? (sumCompleted/sumEnrolled) : 0;\n"
        "    let dRate = sumEnrolled ? (sumDropped/sumEnrolled) : 0;\n"
        "    html += `<p><strong>Combined Enrolled:</strong> ${sumEnrolled}</p>`;\n"
        "    html += `<p><strong>Combined Completed:</strong> ${sumCompleted} (Rate: ${(cRate*100).toFixed(2)}%)</p>`;\n"
        "    html += `<p><strong>Combined Dropped:</strong> ${sumDropped} (Rate: ${(dRate*100).toFixed(2)}%)</p>`;\n"
        "    document.getElementById('trackSeasonOfferingResults').innerHTML = html;\n"
        "}\n"
        "</script>\n"
        "</div></body></html>\n"
    )

    final_html = "".join(html_parts)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Report successfully generated at: {output_html}")


if __name__ == "__main__":
    generate_insights_report(
        input_csv="Data/WR Sankey  copy.csv",
        output_html="report.html"
    )
