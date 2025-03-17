import pandas as pd
import numpy as np
import os
import re
import json
import jinja2
from collections import defaultdict

# Silence future fillna downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

def generate_insights_report(input_csv="Data/Sankey_Pull_March.csv", output_html="report.html"):
    """
    Generates a complete HTML report with all original tables & charts, 
    adapted to the new column names:

      Required Columns:
        Name, Email, Pillar, Track, Season, Level, Session, Group, Non-OIT Group,
        Motivation, Motivation_9_TEXT, Topic Interest, Offering, CMS Center,
        Track Motivation, Track Motivation Other, Track Familiarity, BA Familiarity,
        BA Benefits, BA Application, Benefits of Track, Application of Track,
        Verified Complete, Dropped

      Tables/Sections included:
        1) Overall Stats
        2) Overall Follow-up
        3) By Track
        4) By Track+Season+Offering
        5) Overall by Offering
        6) Combined Track & Season & Level Stats
        7) Motivations
        8) Motivations by Pillar
        9) Topic Interest (Full)
        10) Pillar Insights
        11) Special Interests
        12) BA Familiarity
        13) Benefits of Agile (BA Benefits)
        14) Applications of Agile (BA Application)
        15) Track Familiarity
        16) CMS Center Breakdown
        17) Repeated Enrollments
        18) Repeated Enrollments (By Track)

      Also includes the same 4 charts as before:
        - Seasonal Enrollment Trends
        - Enrollment Trends (Pillar)
        - Average Completion Rate
        - Tracks across Seasons
    """

    # 1) LOAD & VALIDATE
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' does not exist.")
    df = pd.read_csv(input_csv)

    required_cols = [
        "Name", "Email", "Pillar", "Track", "Season", "Level", "Session",
        "Group", "Non-OIT Group", "Motivation", "Motivation_9_TEXT",
        "Topic Interest", "Offering", "CMS Center", "Track Motivation",
        "Track Motivation Other", "Track Familiarity", "BA Familiarity",
        "BA Benefits", "BA Application", "Benefits of Track",
        "Application of Track", "Verified Complete", "Dropped"
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # 2) Normalize "Season"
    def validate_season(season):
        if pd.isna(season):
            return "Unknown Season"
        pattern = r'^(Winter|Spring|Summer|Fall)\s\d{4}$'
        if re.match(pattern, season):
            return season
        else:
            return "Unknown Season"

    df["Season"] = df["Season"].apply(validate_season)

    # Fixed season order
    season_order = [
        "2021", "Winter 2022", "Spring 2022", "Summer 2022", "Fall 2022", "Winter 2023", "Spring 2023", "Summer 2023",
        "Fall 2023", "Winter 2024", "Spring 2024", "Summer 2024",
        "Fall 2024", "Winter 2025", "Spring 2025", "Unknown Season"
    ]
    df["Season"] = pd.Categorical(df["Season"], categories=season_order, ordered=True)

    # Ensure "Track" is categorical
    df["Track"] = df["Track"].fillna("Unknown").astype(str)
    df["Track"] = df["Track"].replace(r'^nan$', "Unknown", regex=True)
    all_tracks_sorted = sorted(df["Track"].unique())
    df["Track"] = pd.Categorical(df["Track"], categories=all_tracks_sorted, ordered=True)

    # Normalize "Level"
    df["Level"] = df["Level"].fillna("Unknown Level").astype(str)
    df["Level"] = df["Level"].replace(r'^nan$', "Unknown Level", regex=True)

    # Convert "Dropped" / "Verified Complete" to booleans
    df["Dropped"] = df["Dropped"].str.lower().map({'yes': True, 'no': False}).fillna(False).astype(bool)
    df["Verified Complete"] = df["Verified Complete"].str.lower().map({'yes': True, 'no': False}).fillna(False).astype(bool)

    # 3) Basic Stats
    total_registrations = len(df)
    total_unique_participants = df["Email"].nunique()
    total_dropped = df["Dropped"].sum()
    total_completed = df["Verified Complete"].sum()
    drop_rate = total_dropped / total_registrations if total_registrations else 0
    completion_rate = total_completed / total_registrations if total_registrations else 0

    # For overall follow-up rate
    enrollments_by_participant = df.groupby("Email", observed=False).agg(
        total_enrollments=("Track", "count"),
        total_completions=("Verified Complete", "sum")
    ).reset_index()
    participants_completed = enrollments_by_participant[enrollments_by_participant["total_completions"] > 0]
    num_completed = len(participants_completed)
    participants_followed_up = participants_completed[participants_completed["total_enrollments"] > 1]
    num_followed_up = len(participants_followed_up)
    overall_follow_up_rate = num_followed_up / num_completed if num_completed else 0

    # (Optional) Combined track categories (e.g. "Cyber Combined", "AI Combined")
    cyber_list = ["Cyber-Hygiene: Advanced Topics", "Cyber-Hygiene: Essentials"]
    ai_list = [
        "AI/ML: AI Applications in Government",
        "AI/ML: Applied Data Science Methods",
        "Artificial Intelligence and Machine Learning",
        "Artificial Intelligence and Machine Learning (AI/ML)",
        "Data Science"
    ]

    def add_combined_track_rows(
        original_df, group_cols, agg_dict, rename_dict=None,
        compute_rates=None, include_unique_participants=False
    ):
        """Adds 'Cyber Combined'/'AI Combined' if grouping by track."""
        grouped = original_df.groupby(group_cols, observed=False).agg(agg_dict).reset_index()

        if include_unique_participants:
            gp_uniq = (
                original_df.groupby(group_cols, observed=False)["Email"].nunique()
                .reset_index().rename(columns={"Email": "unique_participants"})
            )
            grouped = grouped.merge(gp_uniq, on=group_cols, how="left")

        if "Track" in group_cols:
            non_track_cols = [c for c in group_cols if c != "Track"]
            if len(non_track_cols) == 0:
                combos = [{}]
            else:
                combos = grouped[non_track_cols].drop_duplicates().to_dict("records")

            combined_rows = []
            for combo in combos:
                def sum_for_combined(track_list, combined_name):
                    mask = pd.Series([True]*len(original_df))
                    for k,v in combo.items():
                        mask &= (original_df[k] == v)
                    mask &= original_df["Track"].isin(track_list)
                    sub = original_df[mask]
                    if sub.empty:
                        return None
                    row_dict = dict(combo)
                    row_dict["Track"] = combined_name
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

            if combined_rows:
                df_comb = pd.DataFrame(combined_rows)
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

    # Map from Season -> numeric index for follow-up logic
    season2idx = {s: i for i, s in enumerate(season_order)}

    # For each user+track, find earliest completed season
    df_comp = df[df["Verified Complete"]==True].copy()
    df_comp["season_idx"] = df_comp["Season"].map(season2idx)

    earliest_comp = (
        df_comp.groupby(["Email","Track"], observed=False)["season_idx"]
        .min()
        .reset_index()
        .rename(columns={"Track":"TrackForFollowup"})
    )

    # Build user-> list of (track,season_idx) for all enrollments
    df["season_idx"] = df["Season"].map(season2idx)
    df_valid = df.dropna(subset=["season_idx"])
    df_enroll = df_valid[["Email","Track","season_idx"]].copy()

    user_enroll_map = defaultdict(list)
    for _, row in df_enroll.iterrows():
        user = row["Email"]
        track= row["Track"]
        sidx = row["season_idx"]
        user_enroll_map[user].append((track,sidx))

    def new_follow_up_rate(track_name):
        sub = earliest_comp[earliest_comp["TrackForFollowup"] == track_name]
        denom = len(sub)
        if denom==0:
            return "0.00%"
        count_fu=0
        for _, rr in sub.iterrows():
            em = rr["Email"]
            earliest_s = rr["season_idx"]
            enrolls = user_enroll_map[em]
            # Must find a different track in strictly later season
            found_later = any((t_!=track_name) and (sid_>earliest_s) for (t_,sid_) in enrolls)
            if found_later:
                count_fu+=1
        ratio = count_fu/denom
        return f"{ratio:.2%}"

    # 4) By Track aggregator
    def compute_track_rates(df_):
        df_["completion_rate"] = df_["Total Completed"]/df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]/df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x: f"{x:.2%}")
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x: f"{x:.2%}")
        return df_

    track_agg = {"Email":"count","Verified Complete":"sum","Dropped":"sum"}
    track_stats = add_combined_track_rows(
        original_df=df,
        group_cols=["Track"],
        agg_dict=track_agg,
        rename_dict={
            "Email":"Total Enrolled","Verified Complete":"Total Completed","Dropped":"Total Dropped"
        },
        compute_rates=None,
        include_unique_participants=True
    )
    track_stats = compute_track_rates(track_stats)

    # Insert follow-up rate
    f_rates=[]
    for _, row in track_stats.iterrows():
        trk_ = row["Track"]
        fu_  = new_follow_up_rate(trk_)
        f_rates.append(fu_)
    track_stats["Follow-up Rate (Track)"] = f_rates

    # 5) By Track+Season+Offering aggregator, plus Overall by Offering

    # (A) By Track+Season+Offering
    df_tso = (
        df.groupby(["Track","Season","Offering"], observed=False)
          .agg(
              tso_enrolled=("Email","count"),
              tso_completed=("Verified Complete","sum"),
              tso_dropped=("Dropped","sum")
          )
          .reset_index()
    )
    
    # Filter out rows where the total enrolled is zero (i.e. "empty" results)
    df_tso = df_tso[df_tso["tso_enrolled"] > 0]

    # Filter out rows with zero enrollments or unknown offering if desired
    # (If you don't want "Unknown Offering" rows, you can filter them out here.)
    def compute_tso_rates(row):
        e = row["tso_enrolled"]
        c = row["tso_completed"]
        d = row["tso_dropped"]
        cr = c/e if e else 0
        dr = d/e if e else 0
        return pd.Series({
            "Track": row["Track"],
            "Season": row["Season"],
            "Offering": row["Offering"],
            "Total Enrolled": e,
            "Total Completed": c,
            "Total Dropped": d,
            "completion_rate": f"{cr:.2%}",
            "drop_rate": f"{dr:.2%}"
        })
    track_season_offering_stats = df_tso.apply(compute_tso_rates, axis=1).sort_values(
        by=["Track","Season","Offering"]
    ).reset_index(drop=True)

    # (B) Overall by Offering
    df_off = (
        df.groupby("Offering", observed=False)
          .agg(
              total_enrolled=("Email","count"),
              total_completed=("Verified Complete","sum"),
              total_dropped=("Dropped","sum")
          )
          .reset_index()
    )
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
    offering_overall = df_off.apply(compute_offering_rates, axis=1).sort_values("Offering").reset_index(drop=True)

    # 6) Combined Track & Season & Level aggregator
    stl_agg = {"Email":"count","Verified Complete":"sum","Dropped":"sum"}
    season_track_level_stats = add_combined_track_rows(
        original_df=df,
        group_cols=["Season","Track","Level"],
        agg_dict=stl_agg,
        rename_dict={
            "Email":"Total Enrolled",
            "Verified Complete":"Total Completed",
            "Dropped":"Total Dropped"
        },
        compute_rates=None,
        include_unique_participants=True
    )
    def compute_season_track_rates(df_):
        df_["completion_rate"] = df_["Total Completed"] / df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]   / df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x:f"{x:.2%}")
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x:f"{x:.2%}")
        return df_
    def add_all_levels_rows(df_, group_by=["Season","Track"], level_col="Level"):
        new_rows=[]
        for keys,subdf in df_.groupby(group_by, observed=False):
            if not isinstance(keys, tuple):
                keys=(keys,)
            lvls=subdf[level_col].unique()
            if len(lvls)<=1:
                continue
            row_dict={}
            for i,c_ in enumerate(group_by):
                row_dict[c_]= keys[i]
            row_dict[level_col] = "All Levels"
            for c_ in ["Total Enrolled","Total Completed","Total Dropped"]:
                row_dict[c_] = subdf[c_].sum()
            if "unique_participants" in subdf.columns:
                row_dict["unique_participants"] = subdf["unique_participants"].sum()
            new_rows.append(row_dict)
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            return pd.concat([df_, df_new], ignore_index=True)
        else:
            return df_
    season_track_level_stats = add_all_levels_rows(season_track_level_stats, ["Season","Track"], "Level")
    season_track_level_stats = compute_season_track_rates(season_track_level_stats)
    season_track_level_stats = season_track_level_stats.sort_values(
        by=["Track","Level","Season"], ascending=[True,True,True]
    ).reset_index(drop=True)
    for c_ in ["Total Enrolled","Total Completed","Total Dropped"]:
        season_track_level_stats[c_] = season_track_level_stats[c_].fillna(0)

    # 7) Motivations (top 5 per track) & Motivations by Pillar
    # We'll combine "Motivation" + "Motivation_9_TEXT" if present
    def split_motivations(row):
        combined=""
        if pd.notna(row["Motivation"]):
            combined += row["Motivation"]
        if pd.notna(row["Motivation_9_TEXT"]) and row["Motivation_9_TEXT"].strip():
            if combined:
                combined += ", " + row["Motivation_9_TEXT"]
            else:
                combined = row["Motivation_9_TEXT"]
        items=[m.strip() for m in combined.split(",") if m.strip()]
        return items

    df["Motivation List"] = df.apply(split_motivations, axis=1)
    df_mot = df.explode("Motivation List")
    df_mot["Motivation List"] = df_mot["Motivation List"].str.strip()
    df_mot = df_mot[~df_mot["Motivation List"].str.lower().isin(["undefined","unknown","nan"])]
    df_mot = df_mot[df_mot["Motivation List"].str.len()>=1]

    motivation_by_track = (
        df_mot.groupby(["Track","Motivation List"], observed=False)
              .agg(Mentions=("Email","count"))
              .reset_index()
    )
    motivation_by_track = motivation_by_track.sort_values(["Track","Mentions"], ascending=[True,False])
    motivation_by_track = motivation_by_track.groupby("Track", observed=False).head(5).reset_index(drop=True)
    motivation_by_track = motivation_by_track.rename(columns={"Motivation List":"Motivation"})

    # motivations by Pillar
    motivations_trends_by_pillar = (
        df_mot.groupby(["Pillar","Motivation List"], observed=False)
              .size()
              .reset_index(name="Count")
    )
    motivations_trends_by_pillar = motivations_trends_by_pillar.sort_values(["Pillar","Count"], ascending=[True,False])
    motivations_trends_by_pillar = motivations_trends_by_pillar.groupby("Pillar", observed=False).head(5).reset_index(drop=True)
    motivations_trends_by_pillar = motivations_trends_by_pillar.rename(columns={"Motivation List":"Motivation"})
    motivations_trends_by_pillar = motivations_trends_by_pillar[motivations_trends_by_pillar["Count"]!=0]

    # 8) Topic Interest (Full) aggregator
    # We'll mimic the old "topic_interest" aggregator ignoring the pillar
    def rename_dist(d):
        d.rename(columns={"Email":"Count"}, inplace=True)
        return d
    topic_interest = (
        df.groupby(["Track","Topic Interest"], observed=False)["Email"].count()
        .reset_index(name="Count")
    )
    topic_interest = topic_interest[topic_interest["Count"]!=0]
    topic_interest = topic_interest[~topic_interest["Topic Interest"].str.lower().isin(["undefined","unknown","nan"])]
    topic_interest = topic_interest.sort_values(["Track","Topic Interest"]).reset_index(drop=True)

    # 9) Pillar Insights aggregator
    def rename_pillar_cols(df_):
        df_.rename(columns={"Email":"Count"}, inplace=True)
        return df_
    pillar_by_track_agg = (
        df.groupby(["Track","Pillar"], observed=False)["Email"].count()
        .reset_index().rename(columns={"Email":"Count"})
    )
    pillar_by_track_agg = pillar_by_track_agg.sort_values(["Track","Pillar"], ascending=[True,True]).reset_index(drop=True)

    # 10) Special Interests aggregator
    # We'll treat "Topic Interest" as "Special Interests" aggregator if you want 
    # to directly replicate the old table named "Special Interests."
    # Or if you have a separate column named "Special Interests," rename below.
    special_by_track = (
        df.groupby(["Track","Topic Interest"], observed=False)["Email"].count()
        .reset_index().rename(columns={"Email":"Count","Topic Interest":"Special Interests"})
    )
    special_by_track = special_by_track[special_by_track["Count"]!=0]
    special_by_track = special_by_track[~special_by_track["Special Interests"].str.lower().isin(["undefined","unknown","nan"])]
    special_by_track = special_by_track.sort_values(["Track","Special Interests"]).reset_index(drop=True)

    # 11) BA Familiarity, Benefits (Agile), Applications (Agile), each only for "Business Agility" track if you want
    def make_dist_table(col_):
        tmp = (
            df.groupby(["Track", col_], observed=False)["Email"].count()
            .reset_index().rename(columns={"Email": "Count"})
        )
        # Only keep row where Track=="Business Agility" if desired:
        tmp = tmp[tmp["Track"].str.lower().str.strip().str.contains("business agility")]
        if "Track" in tmp.columns:
            tmp.drop(columns=["Track"], inplace=True)
        tmp = tmp.sort_values([col_], ascending=True).reset_index(drop=True)

        # --- FIX: convert the column to string type ---
        tmp[col_] = tmp[col_].astype(str)

        # Now safely filter out 'undefined','unknown','nan'
        tmp = tmp[~tmp[col_].str.lower().isin(["undefined", "unknown", "nan"])]
        
        # *** New step: Remove rows where the Count is 0 ***
        tmp = tmp[tmp["Count"] > 0]

        return tmp


    ba_familiarity_by_track = make_dist_table("Track Familiarity")
    benefits_agile_by_track = make_dist_table("Benefits of Track")
    apps_agile_by_track     = make_dist_table("Application of Track")

    # 12) Track Familiarity aggregator
    track_familiarity = (
        df.groupby(["Track","Track Familiarity"], observed=False)["Email"].count()
        .reset_index().rename(columns={"Email":"Count"})
    )
    track_familiarity = track_familiarity[track_familiarity["Count"]!=0]
    track_familiarity = track_familiarity[~track_familiarity["Track Familiarity"].str.lower().isin(["undefined","unknown","nan"])]
    track_familiarity = track_familiarity.sort_values(["Track","Track Familiarity"]).reset_index(drop=True)

    # 13) CMS Center breakdown aggregator
    def compute_cms_center_rates(df_):
        df_["completion_rate"] = df_["Total Completed"]/df_["Total Enrolled"]
        df_["drop_rate"]       = df_["Total Dropped"]/df_["Total Enrolled"]
        df_["completion_rate"] = df_["completion_rate"].fillna(0).apply(lambda x:f"{x:.2%}")
        df_["drop_rate"]       = df_["drop_rate"].fillna(0).apply(lambda x:f"{x:.2%}")
        return df_

    cms_center_breakdown = (
        df.groupby(["CMS Center","Track"], observed=False)
          .agg({"Email":"count","Verified Complete":"sum","Dropped":"sum"})
          .reset_index().rename(columns={
            "Email":"Total Enrolled","Verified Complete":"Total Completed","Dropped":"Total Dropped"
          })
    )
    cms_center_breakdown = compute_cms_center_rates(cms_center_breakdown)
    cms_center_breakdown = cms_center_breakdown.sort_values(["CMS Center","Track"]).reset_index(drop=True)

    # ---------------- NEW AGGREGATOR: OIT vs Non-OIT ----------------
    # We'll define OIT vs Non-OIT based on "CMS Center"
    df_oit_vs_non_oit = df.copy()
    df_oit_vs_non_oit["OIT vs Non-OIT"] = df_oit_vs_non_oit["CMS Center"].apply(
        lambda x: "OIT" if x == "Office of Information Technology - OIT" else "Non-OIT"
    )

    # Group by Track + Season + OIT/Non-OIT
    oit_non_oit_agg = (
        df_oit_vs_non_oit
        .groupby(["Track", "Season", "OIT vs Non-OIT"], observed=False)["Email"]
        .count()
        .reset_index(name="Count")
        .sort_values(["Track", "Season", "OIT vs Non-OIT"], ascending=True)
        .reset_index(drop=True)
    )
    
    


    # 14) Repeated Enrollments
    repeated_enrollments = df.groupby("Email", observed=False).agg(
        tracks_taken=("Track","nunique"),
        total_enrollments=("Track","count"),
        total_completions=("Verified Complete","sum"),
        total_drops=("Dropped","sum")
    ).reset_index()
    repeated_enrollments = repeated_enrollments[repeated_enrollments["tracks_taken"]>=2]
    multi_track_users = len(repeated_enrollments)

    # 15) Repeated Enrollments (By Track)
    df_times = df.groupby(["Email","Track"], observed=False).size().reset_index(name="times_enrolled")
    df_completed_times = df.groupby(["Email","Track"], observed=False)["Verified Complete"].sum().reset_index(name="times_completed_track")
    repeated_by_email = df.groupby("Email", observed=False).agg(
        total_distinct_tracks=("Track","nunique"),
        total_enrollments=("Track","count"),
        total_completions=("Verified Complete","sum"),
        total_drops=("Dropped","sum")
    ).reset_index()
    repeated_by_track = df_times.merge(df_completed_times, on=["Email","Track"], how="left")
    repeated_enrollments_by_track = repeated_by_track.merge(repeated_by_email, on="Email", how="left").rename(
        columns={
            "times_enrolled":"Times Enrolled in Track",
            "times_completed_track":"Times Completed in Track"
        }
    )
    repeated_enrollments_by_track = repeated_enrollments_by_track.sort_values(["Track","Email"]).reset_index(drop=True)
    max_repeat_count = repeated_enrollments_by_track["Times Enrolled in Track"].max() if not repeated_enrollments_by_track.empty else 1
    max_track_completions = repeated_enrollments_by_track["Times Completed in Track"].max() if not repeated_enrollments_by_track.empty else 0
    
    
    # 16) INCOMPLETE DATA AGGREGATOR
    # -----------------------------------------
    # Identify rows that are missing critical fields or are "incomplete" in a prior season.

    # 1) Compute season_idx for each row (already in df, but let's ensure it's up to date)
    #    We'll also find the maximum "known" season index (ignoring "Unknown Season") for logic below.
    df_incomplete_logic = df.copy()
    
    # Right after: df_incomplete_logic = df.copy()

    # Ensure Full_Cohort exists and is string
    df_incomplete_logic["Full_Cohort"] = df_incomplete_logic["Full_Cohort"].fillna("").astype(str)

    # Exclude anyone with Full_Cohort == 'yes'
    df_incomplete_logic = df_incomplete_logic[
        df_incomplete_logic["Full_Cohort"].str.lower() != "yes"
    ].copy()

    # If the "Unknown Season" category is at the end, let's safely find the max known season index.
    # Filter out 'Unknown Season' to avoid making it the "latest season" by accident.
    valid_seasons_df = df_incomplete_logic[df_incomplete_logic["Season"] != "Unknown Season"]
    if not valid_seasons_df.empty:
        max_known_season_idx = valid_seasons_df["season_idx"].max()
    else:
        # If everything is unknown, define max_known_season_idx as -1 (meaning no known seasons)
        max_known_season_idx = -1

    # 2) Define helper booleans for missing fields
    missing_email = df_incomplete_logic["Email"].isna() | (df_incomplete_logic["Email"].str.strip() == "")
    missing_track = df_incomplete_logic["Track"].isna() | (df_incomplete_logic["Track"].str.lower() == "unknown")
    missing_season = df_incomplete_logic["Season"].isna() | (df_incomplete_logic["Season"] == "Unknown Season")
    # For CMS Center, we'll treat NA/empty or "unknown" as missing
    missing_cms = df_incomplete_logic["CMS Center"].isna() | df_incomplete_logic["CMS Center"].str.lower().isin(["", "nan", "unknown"])

    # 3) Define incomplete if this row is in a past season (season_idx < max_known_season_idx)
    #    and not dropped or completed
    in_past_season = df_incomplete_logic["season_idx"] < max_known_season_idx
    not_dropped_or_completed = (~df_incomplete_logic["Dropped"]) & (~df_incomplete_logic["Verified Complete"])
    missing_past_completion = in_past_season & not_dropped_or_completed
    
    # Count how many *unique participants* are missing completion/dropped in a past season:
    count_incomplete_past_people = df_incomplete_logic.loc[missing_past_completion, "Email"].nunique()


    # 4) Combine all conditions
    incomplete_condition = (missing_email | missing_track | missing_season | missing_cms | missing_past_completion)

    # 5) Keep only rows that are incomplete
    df_incomplete_logic = df_incomplete_logic[incomplete_condition].copy()

    # 6) Build a "Missing Reasons" column to highlight what is missing
    def gather_issues(row):
        reasons = []
        if row["Email"] is None or str(row["Email"]).strip() == "":
            reasons.append("Missing Email")
        if row["Track"].lower() == "unknown":
            reasons.append("Missing Track")
        if row["Season"] == "Unknown Season" or pd.isna(row["Season"]):
            reasons.append("Missing Season")
        cms_val = row["CMS Center"]
        if (pd.isna(cms_val)) or (str(cms_val).lower() in ["", "nan", "unknown"]):
            reasons.append("Missing CMS Center")
        if (row["season_idx"] < max_known_season_idx) and (not row["Verified Complete"]) and (not row["Dropped"]):
            reasons.append("No completion/dropped mark in a past season")
        return "; ".join(reasons)

    df_incomplete_logic["Incomplete Reasons"] = df_incomplete_logic.apply(gather_issues, axis=1)

    # 7) We'll select only the columns we want to display
    incomplete_columns = [
        "Name", "Email", "Track", "Offering", "Season", 
        "Verified Complete", "Dropped", "CMS Center", "Incomplete Reasons"
    ]
    df_incomplete_display = df_incomplete_logic[incomplete_columns].copy()

    # 8) Build the summary counts
    #    We'll count how many rows had each missing piece across the entire dataset df (not just among incomplete rows).
    count_missing_email = missing_email.sum()
    count_missing_track = missing_track.sum()
    count_missing_season = missing_season.sum()
    count_missing_cms = missing_cms.sum()

    # Convert df_incomplete_display to HTML later in the HTML-building section


    # ----------------- CHART DATA (SAME 4 CHARTS) -----------------
    def season_idx_count(df_, col, aggregator='count'):
        grp = df_.groupby("Season", observed=False)[col]
        if aggregator=='count':
            return grp.count().reindex(season_order, fill_value=0).tolist()
        elif aggregator=='sum':
            return grp.sum().reindex(season_order, fill_value=0).tolist()
        else:
            raise ValueError("Aggregator must be 'count' or 'sum'.")

    season_enrollment = season_idx_count(df, "Track", aggregator='count')
    season_completions= season_idx_count(df[df["Verified Complete"]==True], "Verified Complete", aggregator='count')
    season_drops      = season_idx_count(df[df["Dropped"]==True], "Dropped", aggregator='count')

    # Pillar distribution
    enrollment_pillar = df.groupby(["Season","Pillar"], observed=False)["Email"].count().reset_index()
    pivot_pillar = enrollment_pillar.pivot(index="Season", columns="Pillar", values="Email").fillna(0)
    pivot_pillar = pivot_pillar.reindex(season_order).fillna(0)
    pillar_datasets=[]
    i=0
    for c_ in pivot_pillar.columns:
        ds={
            "label": c_,
            "data": pivot_pillar[c_].tolist(),
            "backgroundColor": f"rgba({(i*40)%255}, {(i*80)%255}, {(i*120)%255}, 0.6)",
            "borderColor":     f"rgba({(i*40)%255}, {(i*80)%255}, {(i*120)%255}, 1)",
            "borderWidth":1,
            "fill":False
        }
        i+=1
        pillar_datasets.append(ds)

    # Average completion rate per Season
    completion_rate_per_season = df.groupby("Season", observed=False)["Verified Complete"].mean().reset_index()
    completion_rate_per_season = completion_rate_per_season.set_index("Season").reindex(season_order).fillna(0)
    completion_rate_values = (completion_rate_per_season["Verified Complete"]*100).tolist()

    # Distribution of Tracks across Seasons
    dist_tracks_season = df.groupby(["Season","Track"], observed=False)["Email"].count().reset_index()
    pivot_tracks= dist_tracks_season.pivot(index="Season", columns="Track", values="Email").fillna(0)
    pivot_tracks= pivot_tracks.reindex(season_order).fillna(0)
    distribution_datasets=[]
    i=0
    for c_ in pivot_tracks.columns:
        ds={
            "label": c_,
            "data": pivot_tracks[c_].tolist(),
            "backgroundColor": f"rgba({(i*30)%255}, {(i*60)%255}, {(i*90)%255}, 0.6)",
            "borderColor":     f"rgba({(i*30)%255}, {(i*60)%255}, {(i*90)%255}, 1)",
            "borderWidth":1,
            "fill":False
        }
        i+=1
        distribution_datasets.append(ds)

    # ---------------- BUILD HTML ----------------
    html_parts=[]

    # HEAD + NAV + Overall Stats
    html_parts.append(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8"/>\n'
        "  <title>Participation & Insights Report</title>\n"
        "  <style>\n"
        "    body{ margin:0; font-family:Arial,sans-serif;}\n"
        "    .sidebar{ position:fixed; top:0;left:0; width:220px; height:100%;\n"
        "              overflow-y:auto; background-color:#f4f4f4; padding:20px;\n"
        "              border-right:1px solid #ddd;}\n"
        "    .sidebar h2{margin-top:0;}\n"
        "    .sidebar h3{margin-top:1em; margin-bottom:0.5em; font-size:1.2em;\n"
        "                 border-bottom:1px solid #ccc; padding-bottom:0.3em;}\n"
        "    .sidebar a{ display:block; margin:5px 0; text-decoration:none; color:#333;}\n"
        "    .content{ margin-left:240px; padding:20px;}\n"
        "    h1{ font-size:2em; margin-top:1.5em;}\n"
        "    h2{ font-size:1.75em; margin-top:1.5em;}\n"
        "    h3{ font-size:1.5em; margin-top:1.5em;}\n"
        "    table{ border-collapse:collapse; margin-bottom:1.5em; width:100%;}\n"
        "    table,th,td{ border:1px solid #aaa; padding:8px; text-align:left;}\n"
        "    th{ background-color:#ddd;}\n"
        "    .filter-container{ margin:1em 0;}\n"
        "    .filter-container label{ font-weight:bold; margin-right:0.5em;}\n"
        "    .filter-container select{ margin-right:1em;}\n"
        "    .section-header{ cursor:pointer; display:inline-block; margin-bottom:0.5em;\n"
        "                     padding:8px 12px; background-color:#eee; border:1px solid #ccc;\n"
        "                     border-radius:4px; font-weight:bold; font-size:1.2em;}\n"
        "    .section-content{ border:1px solid #ddd; padding:12px; margin-bottom:1em;\n"
        "                      background-color:#fafafa;}\n"
        "    .chart-section{ margin-bottom:2em;}\n"
        "    .export-btn { \n"
        "      background-color: #4CAF50;\n"
        "      border: none;\n"
        "      color: white;\n"
        "      padding: 8px 16px;\n"
        "      text-align: center;\n"
        "      text-decoration: none;\n"
        "      display: inline-block;\n"
        "      font-size: 14px;\n"
        "      margin: 4px 2px;\n"
        "      cursor: pointer;\n"
        "      border-radius: 4px;\n"
        "    }\n"
        "    .email-modal {\n"
        "      display: none;\n"
        "      position: fixed;\n"
        "      z-index: 1;\n"
        "      left: 0;\n"
        "      top: 0;\n"
        "      width: 100%;\n"
        "      height: 100%;\n"
        "      background-color: rgba(0,0,0,0.4);\n"
        "    }\n"
        "    .email-modal-content {\n"
        "      background-color: #fefefe;\n"
        "      margin: 15% auto;\n"
        "      padding: 20px;\n"
        "      border: 1px solid #888;\n"
        "      width: 80%;\n"
        "      max-width: 500px;\n"
        "    }\n"
        "    .close-modal {\n"
        "      color: #aaa;\n"
        "      float: right;\n"
        "      font-size: 28px;\n"
        "      font-weight: bold;\n"
        "      cursor: pointer;\n"
        "    }\n"
        "    .email-btn {\n"
        "      background-color: #008CBA;\n"
        "      margin-left: 10px;\n"
        "    }\n"
        "  </style>\n"
        '  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n'
        '  <script src="https://cdn.jsdelivr.net/npm/@emailjs/browser@3/dist/email.min.js"></script>\n'
        "</head>\n"
        "<body>\n"
        "<div class='sidebar'>\n"
        "  <h2>Navigation</h2>\n"
        "  <h3>Tables</h3>\n"
        "  <a href='#overall-stats'>Overall Stats</a>\n"
        "  <a href='#overall-followup'>Overall Follow-up</a>\n"
        "  <a href='#by-track-section'>By Track</a>\n"
        "  <a href='#track-season-offering-section'>By Track+Season+Offering</a>\n"
        "  <a href='#offering-overall-section'>Overall by Offering</a>\n"
        "  <a href='#combined-track-season-level-stats-section'>Combined Track &amp; Season &amp; Level Stats</a>\n"
        "  <a href='#motivations-section'>Motivations</a>\n"
        "  <a href='#motivations-by-pillar-section'>Motivations by Pillar</a>\n"
        "  <a href='#topic-interest-section'>Topic Interest (Full)</a>\n"
        "  <a href='#pillar-section'>Pillar Insights</a>\n"
        "  <a href='#special-section'>Special Interests</a>\n"
        "  <a href='#ba-familiarity-section'>BA Familiarity</a>\n"
        "  <a href='#benefits-agile-section'>Benefits of Agile</a>\n"
        "  <a href='#apps-agile-section'>Applications of Agile</a>\n"
        "  <a href='#track-familiarity-section'>Track Familiarity</a>\n"
        "  <a href='#cms-center-section'>CMS Center Breakdown</a>\n"
        "  <a href='#oit-non-oit-section'>OIT vs Non-OIT Breakdown</a>\n"
        "  <a href='#repeated-enrollments-section'>Repeated Enrollments</a>\n"
        "  <a href='#repeated-enrollments-by-track-section'>Repeated Enrollments (By Track)</a>\n"
        "  <a href='#incomplete-data-section'>Incomplete Data</a>\n"
        "  <h3>Charts</h3>\n"
        "  <a href='#charts-section'>Charts Overview</a>\n"
        "  <a href='#season-enrollment-trend-section'>Seasonal Enrollment Trends</a>\n"
        "  <a href='#enrollment-trends-pillar-season-section'>Enrollment Trends (Pillar)</a>\n"
        "  <a href='#average-completion-rate-section'>Average Completion Rate</a>\n"
        "  <a href='#distribution-tracks-season-section'>Tracks across Seasons</a>\n"
        "</div>\n"
        "<div class='content'>\n"
        f"  <h1 id='overall-stats'>Overall Stats</h1>\n"
        f"  <p><strong>Total Registrations:</strong> {total_registrations}</p>\n"
        f"  <p><strong>Total Unique Participants:</strong> {total_unique_participants}</p>\n"
        f"  <p><strong>Total Dropped:</strong> {total_dropped} (Rate: {drop_rate:.2%})</p>\n"
        f"  <p><strong>Total Completed:</strong> {total_completed} (Rate: {completion_rate:.2%})</p>\n"
        "<div id='emailModal' class='email-modal'>\n"
        "  <div class='email-modal-content'>\n"
        "    <span class='close-modal' onclick='closeEmailModal()'>&times;</span>\n"
        "    <h2>Send Table as CSV</h2>\n"
        "    <p>Enter recipient's email address:</p>\n"
        "    <input type='email' id='recipientEmail' placeholder='recipient@example.com'>\n"
        "    <button onclick='sendEmail()' class='export-btn'>Send</button>\n"
        "  </div>\n"
        "</div>\n"
    )

    # Overall follow-up
    html_parts.append(
        "<hr/><h1 id='overall-followup'>Overall Follow-up</h1>\n"
        f"<p>Participants who completed ≥1 track: {num_completed}</p>\n"
        f"<p>Those who enrolled in more than one track: {num_followed_up}</p>\n"
        f"<p><strong>Overall Follow-up Rate:</strong> {overall_follow_up_rate:.2%}</p>\n"
    )

    # ----- By Track -----
    html_parts.append(
        "<hr/><div id='by-track-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('byTrackContent')\">By Track</div>\n"
        "<div class='section-content' id='byTrackContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='byTrackFilter'>Filter by Track:</label>\n"
        "  <select id='byTrackFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    unique_tracks = track_stats["Track"].unique()
    for t_ in unique_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"byTrackTable\", \"track_data.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"byTrackTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='byTrackTable'>\n"
        "<thead><tr>\n"
    )
    for col_ in track_stats.columns:
        html_parts.append(f"  <th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in track_stats.iterrows():
        track_sel = row["Track"]
        html_parts.append(f"<tr data-track='{track_sel}'>")
        for col_ in track_stats.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- By Track+Season+Offering -----
    html_parts.append(
        "<hr/><div id='track-season-offering-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('trackSeasonOfferingContent')\">"
        "By Track+Season+Offering</div>\n"
        "<div class='section-content' id='trackSeasonOfferingContent'>\n"
        "<div class='filter-container'>\n"
        "  <label>Select Track(s):</label>\n"
        "  <select id='tsoTrackFilter' multiple size='5'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    unique_tso_tracks = sorted(track_season_offering_stats["Track"].unique())
    for t_ in unique_tso_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label>Select Season(s):</label>\n"
        "  <select id='tsoSeasonFilter' multiple size='5'>\n"
    )
    for s_ in season_order:
        html_parts.append(f"    <option value='{s_}'>{s_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label>Select Offering(s):</label>\n"
        "  <select id='tsoOfferingFilter' multiple size='5'>\n"
        "    <option value='ALL'>All Offerings</option>\n"
    )
    unique_offs = sorted(track_season_offering_stats["Offering"].unique())
    for off_ in unique_offs:
        html_parts.append(f"    <option value='{off_}'>{off_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button onclick='updateTSOStats()'>Update</button>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"tsoTable\", \"track_season_offering.csv\")' style='margin-left: 10px;'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"tsoTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div id='trackSeasonOfferingResults'></div>\n"
        "</div></div>\n"
    )

    # ----- Overall by Offering -----
    html_parts.append(
        "<hr/><div id='offering-overall-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('offeringOverallContent')\">Overall by Offering</div>\n"
        "<div class='section-content' id='offeringOverallContent'>\n"
        "<div class='filter-container'>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"offeringOverallTable\", \"offering_overall.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"offeringOverallTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='offeringOverallTable'><thead><tr>\n"
    )
    for col_ in offering_overall.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")
    for _, row in offering_overall.iterrows():
        html_parts.append("<tr>")
        for col_ in offering_overall.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- Combined Track & Season & Level Stats -----
    html_parts.append(
        "<hr/><div id='combined-track-season-level-stats-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('combinedTSLContent')\">"
        "Combined Track &amp; Season &amp; Level Stats</div>\n"
        "<div class='section-content' id='combinedTSLContent'>\n"
        "<div class='filter-container'>\n"
        "  <label>Select Track(s) (Ctrl+Click):</label>\n"
        "  <select id='ctslTrackFilter' multiple size='5'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    st_unique_tracks = sorted(season_track_level_stats["Track"].unique())
    for t_ in st_unique_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label>Select Season(s) (Ctrl+Click):</label>\n"
        "  <select id='ctslSeasonFilter' multiple size='5'>\n"
    )
    for s_ in season_order:
        html_parts.append(f"    <option value='{s_}'>{s_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label>Select Level(s) (Ctrl+Click):</label>\n"
        "  <select id='ctslLevelFilter' multiple size='5'>\n"
        "    <option value='ALL'>All Levels</option>\n"
    )
    all_lvls_sorted = sorted(season_track_level_stats["Level"].unique())
    for lv_ in all_lvls_sorted:
        html_parts.append(f"    <option value='{lv_}'>{lv_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button onclick='updateCombinedTSLStats()'>Update</button>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"ctslTable\", \"combined_track_season_level.csv\")' style='margin-left: 10px;'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"ctslTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div id='combinedTSLResults'></div>\n"
        "</div></div>\n"
    )

    # ----- Motivations (top 5 per track) -----
    html_parts.append(
        "<hr/><div id='motivations-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('motivationsContent')\">Motivations</div>\n"
        "<div class='section-content' id='motivationsContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='motivationsFilter'>Filter by Track:</label>\n"
        "  <select id='motivationsFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    mot_tracks = sorted(motivation_by_track["Track"].unique())
    for t_ in mot_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"motivationsTable\", \"motivations.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"motivationsTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='motivationsTable'><thead><tr>\n"
    )
    for col_ in motivation_by_track.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in motivation_by_track.iterrows():
        trk_ = row["Track"]
        if row["Mentions"]==0:
            continue
        html_parts.append(f"<tr data-track='{trk_}'>")
        for col_ in motivation_by_track.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- Motivations by Pillar (Top 5) -----
    html_parts.append(
        "<hr/><div id='motivations-by-pillar-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('motivsPillarContent')\">Motivations by Pillar</div>\n"
        "<div class='section-content' id='motivsPillarContent'>\n"
        "<div class='filter-container'>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"motivationsPillarTable\", \"motivations_by_pillar.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"motivationsPillarTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='motivationsPillarTable'><thead><tr><th>Pillar</th><th>Motivation</th><th>Count</th></tr></thead><tbody>\n"
    )
    for _, row in motivations_trends_by_pillar.iterrows():
        if row["Count"]==0:
            continue
        html_parts.append(
            f"<tr><td>{row['Pillar']}</td>"
            f"<td>{row['Motivation']}</td>"
            f"<td>{row['Count']}</td></tr>\n"
        )
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- Topic Interest (Full) -----
    html_parts.append(
        "<hr/><div id='topic-interest-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('topicInterestContent')\">Topic Interest (Full)</div>\n"
        "<div class='section-content' id='topicInterestContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='topicInterestFilter'>Filter by Track:</label>\n"
        "  <select id='topicInterestFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    ti_tracks = sorted(topic_interest["Track"].unique())
    for t_ in ti_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"topicInterestTable\", \"topic_interest.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"topicInterestTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='topicInterestTable'><thead><tr>\n"
    )
    for col_ in topic_interest.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in topic_interest.iterrows():
        trk_ = row["Track"]
        if row["Count"]==0:
            continue
        html_parts.append(f"<tr data-track='{trk_}'>")
        for col_ in topic_interest.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")

    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- Pillar Insights -----
    html_parts.append(
        "<hr/><div id='pillar-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('pillarContent')\">Pillar Insights</div>\n"
        "<div class='section-content' id='pillarContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='pillarFilter'>Filter by Track:</label>\n"
        "  <select id='pillarFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    unique_pillar_tracks = sorted(pillar_by_track_agg["Track"].unique())
    for t_ in unique_pillar_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"pillarTable\", \"pillar_insights.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"pillarTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='pillarTable'><thead><tr>\n"
    )
    for col_ in pillar_by_track_agg.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in pillar_by_track_agg.iterrows():
        trk_ = row["Track"]
        if row["Count"]==0:
            continue
        html_parts.append(f"<tr data-track='{trk_}'>")
        for col_ in pillar_by_track_agg.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- Special Interests -----
    html_parts.append(
        "<hr/><div id='special-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('specialContent')\">Special Interests</div>\n"
        "<div class='section-content' id='specialContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='specialFilter'>Filter by Track:</label>\n"
        "  <select id='specialFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    unique_spec_tracks = sorted(special_by_track["Track"].unique())
    for t_ in unique_spec_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"specialTable\", \"special_interests.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"specialTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='specialTable'><thead><tr>\n"
    )
    for col_ in special_by_track.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in special_by_track.iterrows():
        trk_ = row["Track"]
        if row["Count"]==0:
            continue
        html_parts.append(f"<tr data-track='{trk_}'>")
        for col_ in special_by_track.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- BA Familiarity, Benefits, Applications (for "Business Agility") -----
    # BA Familiarity
    html_parts.append(
        "<hr/><div id='ba-familiarity-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('baFamiliarityContent')\">BA Familiarity (Business Agility)</div>\n"
        "<div class='section-content' id='baFamiliarityContent'>\n"
        "<div class='filter-container'>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"baFamiliarityTable\", \"ba_familiarity.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"baFamiliarityTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div class='table-container' id='baFamiliarityTable'>\n"
        f"{ba_familiarity_by_track.to_html(index=False)}\n"
        "</div></div></div>\n"
    )
    # Benefits of Agile => "BA Benefits"
    html_parts.append(
        "<hr/><div id='benefits-agile-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('benefitsAgileContent')\">Benefits of Agile (Business Agility)</div>\n"
        "<div class='section-content' id='benefitsAgileContent'>\n"
        "<div class='filter-container'>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"benefitsAgileTable\", \"benefits_agile.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"benefitsAgileTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div class='table-container' id='benefitsAgileTable'>\n"
        f"{benefits_agile_by_track.to_html(index=False)}\n"
        "</div></div></div>\n"
    )
    # Applications of Agile => "BA Application"
    html_parts.append(
        "<hr/><div id='apps-agile-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('appsAgileContent')\">Applications of Agile (Business Agility)</div>\n"
        "<div class='section-content' id='appsAgileContent'>\n"
        "<div class='filter-container'>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"appsAgileTable\", \"applications_agile.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"appsAgileTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div class='table-container' id='appsAgileTable'>\n"
        f"{apps_agile_by_track.to_html(index=False)}\n"
        "</div></div></div>\n"
    )

    # ----- Track Familiarity -----
    html_parts.append(
        "<hr/><div id='track-familiarity-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('trackFamiliarityContent')\">Track Familiarity</div>\n"
        "<div class='section-content' id='trackFamiliarityContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='trackFamiliarityFilter'>Filter by Track:</label>\n"
        "  <select id='trackFamiliarityFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    unique_fam_tracks = sorted(track_familiarity["Track"].unique())
    for t_ in unique_fam_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"trackFamiliarityTable\", \"track_familiarity.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"trackFamiliarityTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='trackFamiliarityTable'><thead><tr>\n"
    )
    for col_ in track_familiarity.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in track_familiarity.iterrows():
        trk_ = row["Track"]
        if row["Count"]==0:
            continue
        html_parts.append(f"<tr data-track='{trk_}'>")
        for col_ in track_familiarity.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")

    # ----- CMS Center Breakdown -----
    html_parts.append(
        "<hr/><div id='cms-center-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('cmsCenterContent')\">CMS Center Breakdown</div>\n"
        "<div class='section-content' id='cmsCenterContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='cmsCenterTrackFilter'>Filter by Track:</label>\n"
        "  <select id='cmsCenterTrackFilter'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    uniq_cms_tracks = sorted(cms_center_breakdown["Track"].unique())
    for t_ in uniq_cms_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    unique_centers = sorted(cms_center_breakdown["CMS Center"].unique())
    html_parts.append(
        "  </select>\n"
        "  <label for='cmsCenterFilterCenter'>Filter by CMS Center:</label>\n"
        "  <select id='cmsCenterFilterCenter'>\n"
        "    <option value='ALL'>All Centers</option>\n"
    )
    for c_ in unique_centers:
        html_parts.append(f"    <option value='{c_}'>{c_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"cmsCenterTable\", \"cms_center_breakdown.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"cmsCenterTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<table id='cmsCenterTable'><thead><tr>\n"
    )
    for col_ in cms_center_breakdown.columns:
        html_parts.append(f"<th>{col_}</th>")
    html_parts.append("</tr></thead><tbody>\n")

    for _, row in cms_center_breakdown.iterrows():
        track_ = row["Track"]
        center_= row["CMS Center"]
        html_parts.append(f"<tr data-track='{track_}' data-center='{center_}'>")
        for col_ in cms_center_breakdown.columns:
            html_parts.append(f"<td>{row[col_]}</td>")
        html_parts.append("</tr>\n")
    html_parts.append("</tbody></table>\n</div></div>\n")
    
    html_parts.append(
    "<hr/><div id='oit-non-oit-section'>\n"
    "<div class='section-header' onclick=\"toggleSection('oitNonOitContent')\">"
    "OIT vs Non-OIT Breakdown</div>\n"
    "<div class='section-content' id='oitNonOitContent'>\n"

    "<div class='filter-container'>\n"
    "  <label for='oitTrackFilter'>Select Track(s) (Ctrl+Click):</label>\n"
    "  <select id='oitTrackFilter' multiple size='5'>\n"
    "    <option value='ALL'>All Tracks</option>\n"
)

    # Add track <option> items
    unique_tracks_for_oit = sorted(oit_non_oit_agg["Track"].unique())
    for t_ in unique_tracks_for_oit:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")

    html_parts.append(
        "  </select>\n"
        "  <label for='oitSeasonFilter'>Select Season(s) (Ctrl+Click):</label>\n"
        "  <select id='oitSeasonFilter' multiple size='5'>\n"
    )

    # Add season <option> items
    for s_ in season_order:
        html_parts.append(f"    <option value='{s_}'>{s_}</option>\n")

    html_parts.append(
        "  </select>\n"
        "  <button onclick='updateOITBreakdown()'>Update</button>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"oitTable\", \"oit_breakdown.csv\")' style='margin-left: 10px;'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"oitTable\")'>Email CSV</button>\n"
        "</div>\n"  # close .filter-container

        "<div id='oitBreakdownResults'></div>\n"
        "</div></div>\n"
    )


    # ----- Repeated Enrollments -----
    html_parts.append(
        f"<hr/><div id='repeated-enrollments-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('repeatedEnrollmentsContent')\">Repeated Enrollments</div>\n"
        f"<div class='section-content' id='repeatedEnrollmentsContent'>\n"
        f"<p>Number of participants who enrolled in multiple distinct tracks: {multi_track_users}</p>\n"
        f"{repeated_enrollments.to_html(index=False)}\n"
        "</div></div>\n"
    )

    # ----- Repeated Enrollments (By Track) -----
    html_parts.append(
        "<hr/><div id='repeated-enrollments-by-track-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('repeatedByTrackContent')\">Repeated Enrollments (By Track)</div>\n"
        "<div class='section-content' id='repeatedByTrackContent'>\n"
        "<div class='filter-container'>\n"
        "  <label for='repeatedByTrackSelect'>Select Track(s) (Ctrl+Click):</label>\n"
        "  <select id='repeatedByTrackSelect' multiple size='5'>\n"
        "    <option value='ALL'>All Tracks</option>\n"
    )
    rep_tracks = sorted(repeated_enrollments_by_track["Track"].unique())
    for t_ in rep_tracks:
        html_parts.append(f"    <option value='{t_}'>{t_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label for='repeatCountSelect'>Select Repeat Count(s) (Ctrl+Click):</label>\n"
        "  <select id='repeatCountSelect' multiple size='5'>\n"
    )
    for c_ in range(1, max_repeat_count+1):
        html_parts.append(f"    <option value='{c_}'>{c_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <label for='completionCountSelect'>Select Completion Count(s) (Ctrl+Click):</label>\n"
        "  <select id='completionCountSelect' multiple size='5'>\n"
    )
    for c_ in range(0, max_track_completions+1):
        html_parts.append(f"    <option value='{c_}'>{c_}</option>\n")
    html_parts.append(
        "  </select>\n"
        "  <button onclick='updateRepeatedByTrackStats()'>Update</button>\n"
        "  <button class='export-btn' onclick='exportTableToCSV(\"repeatedByTrackTable\", \"repeated_enrollments_by_track.csv\")'>Export to CSV</button>\n"
        "  <button class='export-btn email-btn' onclick='showEmailModal(\"repeatedByTrackTable\")'>Email CSV</button>\n"
        "</div>\n"
        "<div id='repeatedByTrackResults'></div>\n"
        "</div></div>\n"
    )
    
    # =============== INCOMPLETE DATA SECTION ===============

    # 1) The aggregator logic must have produced df_incomplete_display already.
    #    It also must have 9 columns in this exact order (or very similar):
    #    [Name, Email, Track, Offering, Season, Verified Complete, Dropped, CMS Center, Incomplete Reasons]

    # 2) Define a small function to highlight the missing cells in red.
    def highlight_incomplete_cells(row):
        """
        row is a single row (Series). We'll return a list of CSS strings,
        one for each cell, e.g. ["background-color: #ffbaba;", "", "", ...]
        to highlight only missing or incomplete items.
        """
        # We'll build a list of styles for each of the columns in df_incomplete_display
        style_list = [""] * len(row)  # default: no style

        # Check each column by name:
        # (Adjust these conditions to match how you define "missing" or "unknown")
        if pd.isna(row["Email"]) or str(row["Email"]).strip() == "":
            idx = row.index.get_loc("Email")
            style_list[idx] = "background-color: #ffcccc;"

        if str(row["Track"]).lower() == "unknown":
            idx = row.index.get_loc("Track")
            style_list[idx] = "background-color: #ffcccc;"

        if pd.isna(row["Season"]) or row["Season"] == "Unknown Season":
            idx = row.index.get_loc("Season")
            style_list[idx] = "background-color: #ffcccc;"

        cms_val = row["CMS Center"]
        if pd.isna(cms_val) or str(cms_val).lower() in ["", "nan", "unknown"]:
            idx = row.index.get_loc("CMS Center")
            style_list[idx] = "background-color: #ffcccc;"

        # Finally, highlight Verified Complete / Dropped if the "Incomplete Reasons"
        # indicates "No completion/dropped mark in a past season"
        if "No completion/dropped mark in a past season" in str(row["Incomplete Reasons"]):
            idx_complete = row.index.get_loc("Verified Complete")
            idx_dropped  = row.index.get_loc("Dropped")
            style_list[idx_complete] = "background-color: #ffcccc;"
            style_list[idx_dropped]  = "background-color: #ffcccc;"

        return style_list

    # 3) Apply the style function
    df_incomplete_styled = df_incomplete_display.style.apply(highlight_incomplete_cells, axis=1)

    # 4) Convert it to HTML
    df_incomplete_html = df_incomplete_styled.to_html(index=False, justify="left")

    # 5) Build the new section in html_parts (including the side nav link you added earlier)
    html_parts.append(
        "<hr/><div id='incomplete-data-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('incompleteDataContent')\">"
        "Incomplete Data</div>\n"
        "<div class='section-content' id='incompleteDataContent'>\n"
        "<p>The table below shows rows that are missing key fields (e.g. email, track, season, CMS center) "
        "or are in a past season but never marked dropped or completed.</p>\n"
    )

    # You can also append your summary counts (e.g. how many rows missing email, track, etc.)
    html_parts.append(f"<p><strong>Total Rows Missing Email:</strong> {count_missing_email}</p>")
    html_parts.append(f"<p><strong>Total Rows Missing Track:</strong> {count_missing_track}</p>")
    html_parts.append(f"<p><strong>Total Rows Missing Season:</strong> {count_missing_season}</p>")
    html_parts.append(f"<p><strong>Total Rows Missing CMS Center:</strong> {count_missing_cms}</p>")
    html_parts.append(f"<p><strong>Participants in a previous season without 'Dropped' or 'Completed':</strong> {count_incomplete_past_people}</p>")


    # Finally, append the highlighted table
    html_parts.append(df_incomplete_html)

    # Close section
    html_parts.append("</div></div>\n")



    # ----- Charts Section -----
    html_parts.append(
        "<hr/><div id='charts-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('chartsContent')\">Charts (Click to Minimize/Expand)</div>\n"
        "<div class='section-content' id='chartsContent'>\n"

        "<div class='chart-section' id='season-enrollment-trend-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('seasonEnrollmentTrendContent')\">Seasonal Enrollment Trends</div>\n"
        "<div class='section-content' id='seasonEnrollmentTrendContent'>\n"
        "<canvas id='seasonEnrollmentChart' width='800' height='400'></canvas>\n"
        "</div></div>\n"

        "<div class='chart-section' id='enrollment-trends-pillar-season-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('enrollmentTrendsPillarSeasonContent')\">Enrollment Trends (Pillar)</div>\n"
        "<div class='section-content' id='enrollmentTrendsPillarSeasonContent'>\n"
        "<canvas id='pillarTrendsChart' width='800' height='400'></canvas>\n"
        "</div></div>\n"

        "<div class='chart-section' id='average-completion-rate-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('avgCompletionRateContent')\">Average Completion Rate</div>\n"
        "<div class='section-content' id='avgCompletionRateContent'>\n"
        "<canvas id='avgCompletionChart' width='800' height='400'></canvas>\n"
        "</div></div>\n"

        "<div class='chart-section' id='distribution-tracks-season-section'>\n"
        "<div class='section-header' onclick=\"toggleSection('distributionTracksSeasonContent')\">Tracks across Seasons</div>\n"
        "<div class='section-content' id='distributionTracksSeasonContent'>\n"
        "<canvas id='trackDistChart' width='800' height='400'></canvas>\n"
        "</div></div>\n"

        "</div></div>\n"
    )

    # JSON data for chart building + aggregator table usage
    season_order_json            = json.dumps(season_order)
    season_enrollment_json       = json.dumps(season_enrollment)
    season_completions_json      = json.dumps(season_completions)
    season_drops_json            = json.dumps(season_drops)
    pillar_datasets_json         = json.dumps(pillar_datasets)
    completion_rate_values_json  = json.dumps(completion_rate_values)
    distribution_datasets_json   = json.dumps(distribution_datasets)

    # By Track+Season+Offering aggregator data
    track_season_offering_records = track_season_offering_stats.to_dict("records")
    tso_data_json = json.dumps(track_season_offering_records)

    # For combined aggregator table
    stl_records = season_track_level_stats.to_dict("records")
    stl_json = json.dumps(stl_records)

    # For repeated enrollments by track
    repeated_by_track_records = repeated_enrollments_by_track.to_dict("records")
    repeated_by_track_json = json.dumps(repeated_by_track_records)
    
    oit_non_oit_records = oit_non_oit_agg.to_dict("records")
    oit_data_json = json.dumps(oit_non_oit_records)

    

    # JS for interactivity
    html_parts.append(
        "<script>\n"
        "let currentTableId = '';\n\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "  // Initialize EmailJS\n"
        "  emailjs.init('_APatchKicG-TbQ84');\n\n"
        "  // Email modal functions\n"
        "  window.showEmailModal = function(tableId) {\n"
        "    currentTableId = tableId;\n"
        "    document.getElementById('emailModal').style.display = 'block';\n"
        "  };\n\n"
        "  window.closeEmailModal = function() {\n"
        "    document.getElementById('emailModal').style.display = 'none';\n"
        "    document.getElementById('recipientEmail').value = '';\n"
        "  };\n\n"
        "  window.sendEmail = async function() {\n"
        "    const recipientEmail = document.getElementById('recipientEmail').value;\n"
        "    if (!recipientEmail) {\n"
        "      alert('Please enter a recipient email address');\n"
        "      return;\n"
        "    }\n\n"
        "    try {\n"
        "      const table = document.getElementById(currentTableId);\n"
        "      const rows = Array.from(table.querySelectorAll('tr:not([style*=\"display: none\"])'));\n"
        "      let csvContent = '';\n\n"
        "      // Get headers\n"
        "      const headers = Array.from(rows[0].querySelectorAll('th'))\n"
        "        .map(header => '\"' + header.textContent.replace(/\"/g, '\"\"') + '\"');\n"
        "      csvContent += headers.join(',') + '\\n';\n\n"
        "      // Get rows\n"
        "      rows.slice(1).forEach(row => {\n"
        "        const cells = Array.from(row.querySelectorAll('td'))\n"
        "          .map(cell => '\"' + cell.textContent.replace(/\"/g, '\"\"') + '\"');\n"
        "        csvContent += cells.join(',') + '\\n';\n"
        "      });\n\n"
        "      const instructions = `\n"
        "For Windows:\n"
        "1. Open Notepad\n"
        "2. Copy and paste the CSV data below into Notepad\n"
        "3. Click File -> Save As\n"
        "4. Set 'Save as type' to 'All Files (*.*)'\n"
        "5. Name your file with .csv extension (e.g., ${currentTableId}.csv)\n"
        "6. Click Save\n\n"
        "For Mac:\n"
        "1. Open TextEdit\n"
        "2. Click Format -> Make Plain Text (or press Shift+Command+T)\n"
        "3. Copy and paste the CSV data below into TextEdit\n"
        "4. Click File -> Save\n"
        "5. Add .csv extension to the filename (e.g., ${currentTableId}.csv)\n"
        "6. Click Save\n\n"
        "CSV Data (copy everything below this line):\n"
        "----------------------------------------\n`;\n"
        "      const response = await emailjs.send(\n"
        "        'service_pu6xmbj',\n"
        "        'template_7dlxryl',\n"
        "        {\n"
        "          to_email: recipientEmail,\n"
        "          table_name: currentTableId,\n"
        "          instructions: instructions,\n"
        "          csv_data: csvContent\n"
        "        }\n"
        "      );\n\n"
        "      alert('Email sent successfully!');\n"
        "      closeEmailModal();\n"
        "    } catch (err) {\n"
        "      console.error('Failed to send email:', err);\n"
        "      alert('Failed to send email. Please try again.');\n"
        "    }\n"
        "  };\n\n"
        "});\n\n"
        f"var oitNonOitData = {oit_data_json};\n" +
        "function toggleSection(contentId){\n"
        "  const el=document.getElementById(contentId);\n"
        "  el.style.display=(el.style.display==='none'?'block':'none');\n"
        "}\n\n"
        "// By Track filter\n"
        "document.addEventListener('DOMContentLoaded',function(){\n"
        "  let trackSel = document.getElementById('byTrackFilter');\n"
        "  let table = document.getElementById('byTrackTable');\n"
        "  let rows = table.querySelectorAll('tbody tr');\n"
        "  trackSel.addEventListener('change',function(){\n"
        "    let chosen=trackSel.value;\n"
        "    rows.forEach(r=>{\n"
        "      let rowTrack=r.getAttribute('data-track');\n"
        "      r.style.display=(chosen==='ALL'||rowTrack===chosen)?'':'none';\n"
        "    });\n"
        "  });\n"
        "});\n\n"
        "// Generic track-based filter for some tables\n"
        "function addTrackFilterListener(tableId, dropdownId){\n"
        "  let dd=document.getElementById(dropdownId);\n"
        "  dd.addEventListener('change',function(){\n"
        "    let chosen=dd.value;\n"
        "    let table=document.getElementById(tableId);\n"
        "    let rows=table.querySelectorAll('tbody tr');\n"
        "    rows.forEach(r=>{\n"
        "      let rowTrack=r.getAttribute('data-track');\n"
        "      r.style.display=(chosen==='ALL'||rowTrack===chosen)?'':'none';\n"
        "    });\n"
        "  });\n"
        "}\n"
        f"var tsoData={tso_data_json};\n"
        f"var seasonOrder={season_order_json};\n\n"
        "function updateTSOStats(){\n"
        "  let trackSel = document.getElementById('tsoTrackFilter');\n"
        "  let chosenTracks = Array.from(trackSel.selectedOptions).map(opt => opt.value);\n"
        "  if(chosenTracks.includes('ALL') || chosenTracks.length === 0){\n"
        "    chosenTracks = null;\n"
        "  }\n\n"
        "  let seasonSel = document.getElementById('tsoSeasonFilter');\n"
        "  let chosenSeasons = Array.from(seasonSel.selectedOptions).map(opt => opt.value);\n"
        "  if(chosenSeasons.length === 0){\n"
        "    chosenSeasons = seasonOrder;\n"
        "  }\n\n"
        "  let offeringSel = document.getElementById('tsoOfferingFilter');\n"
        "  let chosenOfferings = Array.from(offeringSel.selectedOptions).map(opt => opt.value);\n"
        "  if(chosenOfferings.includes('ALL') || chosenOfferings.length === 0){\n"
        "    chosenOfferings = null;\n"
        "  }\n\n"
        "  let filtered = tsoData.filter(row => {\n"
        "    let trackMatch = chosenTracks ? chosenTracks.includes(row['Track']) : true;\n"
        "    let seasonMatch = chosenSeasons.includes(row['Season']);\n"
        "    let offeringMatch = chosenOfferings ? chosenOfferings.includes(row['Offering']) : true;\n"
        "    return trackMatch && seasonMatch && offeringMatch;\n"
        "  });\n\n"
        "  let html = \"<table id='tsoTable'><thead><tr>\" +\n"
        "             \"<th>Track</th><th>Season</th><th>Offering</th>\" +\n"
        "             \"<th>Total Enrolled</th><th>Total Completed</th><th>Total Dropped</th>\" +\n"
        "             \"<th>Completion Rate</th><th>Drop Rate</th>\" +\n"
        "             \"</tr></thead><tbody>\";\n\n"
        "  filtered.forEach(row => {\n"
        "    html += `<tr><td>${row['Track']}</td><td>${row['Season']}</td><td>${row['Offering']}</td>` +\n"
        "            `<td>${row['Total Enrolled']}</td><td>${row['Total Completed']}</td>` +\n"
        "            `<td>${row['Total Dropped']}</td><td>${row['completion_rate']}</td>` +\n"
        "            `<td>${row['drop_rate']}</td></tr>`;\n"
        "  });\n\n"
        "  html += \"</tbody></table>\";\n"
        "  document.getElementById('trackSeasonOfferingResults').innerHTML = html;\n"
        "}\n"
        "// For Motivations, Topic Interest, Pillar, Special, TrackFamiliarity\n"
        "addTrackFilterListener('motivationsTable','motivationsFilter');\n"
        "addTrackFilterListener('topicInterestTable','topicInterestFilter');\n"
        "addTrackFilterListener('pillarTable','pillarFilter');\n"
        "addTrackFilterListener('specialTable','specialFilter');\n"
        "addTrackFilterListener('trackFamiliarityTable','trackFamiliarityFilter');\n\n"
        "// CMS Center filter\n"
        "document.addEventListener('DOMContentLoaded',function(){\n"
        "  let trkSel=document.getElementById('cmsCenterTrackFilter');\n"
        "  let ctrSel=document.getElementById('cmsCenterFilterCenter');\n"
        "  let table=document.getElementById('cmsCenterTable');\n"
        "  let rows=table.querySelectorAll('tbody tr');\n"
        "  function filterCMS(){\n"
        "    let chosenTrack=trkSel.value;\n"
        "    let chosenCenter=ctrSel.value;\n"
        "    rows.forEach(r=>{\n"
        "      let rowT=r.getAttribute('data-track');\n"
        "      let rowC=r.getAttribute('data-center');\n"
        "      let matchT=(chosenTrack==='ALL'||rowT===chosenTrack);\n"
        "      let matchC=(chosenCenter==='ALL'||rowC===chosenCenter);\n"
        "      r.style.display=(matchT&&matchC)?'':'none';\n"
        "    });\n"
        "  }\n"
        "  trkSel.addEventListener('change',filterCMS);\n"
        "  ctrSel.addEventListener('change',filterCMS);\n"
        "});\n\n"
        "// Combined aggregator for (Track+Season+Level)\n"
        f"var stlData={stl_json};\n"
        f"var seasonOrder={season_order_json};\n"
        "function updateCombinedTSLStats(){\n"
        "  let trkSel=document.getElementById('ctslTrackFilter');\n"
        "  let chosenTracks=Array.from(trkSel.selectedOptions).map(opt=>opt.value);\n"
        "  if(chosenTracks.includes('ALL')||chosenTracks.length===0){chosenTracks=null;}\n"
        "  let seaSel=document.getElementById('ctslSeasonFilter');\n"
        "  let chosenSeasons=Array.from(seaSel.selectedOptions).map(opt=>opt.value);\n"
        "  if(chosenSeasons.length===0){chosenSeasons=seasonOrder;}\n"
        "  let lvlSel=document.getElementById('ctslLevelFilter');\n"
        "  let chosenLevels=Array.from(lvlSel.selectedOptions).map(opt=>opt.value);\n"
        "  if(chosenLevels.includes('ALL')||chosenLevels.length===0){chosenLevels=null;}\n"
        "  let filtered=stlData.filter(r=>{\n"
        "    let trackMatch=true;\n"
        "    if(chosenTracks){ trackMatch=chosenTracks.includes(r['Track']);}\n"
        "    let seasonMatch=chosenSeasons.includes(r['Season']);\n"
        "    let levelMatch=true;\n"
        "    if(chosenLevels){ levelMatch=chosenLevels.includes(r['Level']);}\n"
        "    return trackMatch&&seasonMatch&&levelMatch;\n"
        "  });\n"
        "  filtered=filtered.filter(r=> parseFloat(r['Total Enrolled'])>0);\n"
        "  if(filtered.length===0){\n"
        "    document.getElementById('combinedTSLResults').innerHTML='<p>No data found.</p>';\n"
        "    return;\n"
        "  }\n"
        "  let html=\"<table id='ctslTable'><thead><tr>"
        "<th>Season</th><th>Track</th><th>Level</th>"
        "<th>Total Enrolled</th><th>Total Completed</th><th>Total Dropped</th>"
        "<th>completion_rate</th><th>drop_rate</th><th>unique_participants</th>"
        "</tr></thead><tbody>\";\n"
        "  let sumEn=0,sumComp=0,sumDrop=0;\n"
        "  filtered.forEach(row=>{\n"
        "    html+=`<tr><td>${row['Season']}</td><td>${row['Track']}</td><td>${row['Level']}</td>`;\n"
        "    html+=`<td>${row['Total Enrolled']}</td><td>${row['Total Completed']}</td>`;\n"
        "    html+=`<td>${row['Total Dropped']}</td>`;\n"
        "    html+=`<td>${row['completion_rate']||'0.00%'}</td><td>${row['drop_rate']||'0.00%'}</td>`;\n"
        "    html+=`<td>${row['unique_participants']||''}</td></tr>`;\n"
        "    let e=parseFloat(row['Total Enrolled'])||0;\n"
        "    let c=parseFloat(row['Total Completed'])||0;\n"
        "    let d=parseFloat(row['Total Dropped'])||0;\n"
        "    sumEn+=e; sumComp+=c; sumDrop+=d;\n"
        "  });\n"
        "  html+='</tbody></table>';\n"
        "  let cRate=(sumEn? (sumComp/sumEn):0)*100;\n"
        "  let dRate=(sumEn? (sumDrop/sumEn):0)*100;\n"
        "  html+=`<p><strong>Combined Enrolled:</strong> ${sumEn}</p>`;\n"
        "  html+=`<p><strong>Combined Completed:</strong> ${sumComp} (Rate: ${cRate.toFixed(2)}%)</p>`;\n"
        "  html+=`<p><strong>Combined Dropped:</strong> ${sumDrop} (Rate: ${dRate.toFixed(2)}%)</p>`;\n"
        "  document.getElementById('combinedTSLResults').innerHTML=html;\n"
        "}\n\n"
        "// Repeated Enrollments (By Track)\n"
        f"var repeatedByTrackData={repeated_by_track_json};\n"
        "function updateRepeatedByTrackStats(){\n"
        "  let tSel=document.getElementById('repeatedByTrackSelect');\n"
        "  let chosenT=Array.from(tSel.selectedOptions).map(o=>o.value);\n"
        "  if(chosenT.includes('ALL')||chosenT.length===0){chosenT=null;}\n"
        "  let repSel=document.getElementById('repeatCountSelect');\n"
        "  let chosenR=Array.from(repSel.selectedOptions).map(o=>parseInt(o.value));\n"
        "  if(chosenR.length===0){chosenR=null;}\n"
        "  let compSel=document.getElementById('completionCountSelect');\n"
        "  let chosenC=Array.from(compSel.selectedOptions).map(o=>parseInt(o.value));\n"
        "  if(chosenC.length===0){chosenC=null;}\n"
        "  let filtered=repeatedByTrackData;\n"
        "  if(chosenT){ filtered=filtered.filter(r=>chosenT.includes(r['Track']));}\n"
        "  if(chosenR){ filtered=filtered.filter(r=>chosenR.includes(r['Times Enrolled in Track']));}\n"
        "  if(chosenC){ filtered=filtered.filter(r=>chosenC.includes(r['Times Completed in Track']));}\n"
        "  if(filtered.length===0){\n"
        "    document.getElementById('repeatedByTrackResults').innerHTML='<p>No data found.</p>';\n"
        "    return;\n"
        "  }\n"
        "  let uniqueEmails=new Set(filtered.map(x=>x['Email']));\n"
        "  let countUnique=uniqueEmails.size;\n"
        "  let html=`<p>Number of participants matching filters: <strong>${countUnique}</strong></p>`;\n"
        "  html+='<table id=\"repeatedByTrackTable\"><thead><tr>';\n"
        "  let columns=['Email','Track','Times Enrolled in Track','Times Completed in Track','total_distinct_tracks','total_enrollments','total_completions','total_drops'];\n"
        "  columns.forEach(c=>{html+=`<th>${c}</th>`;});\n"
        "  html+='</tr></thead><tbody>';\n"
        "  filtered.forEach(row=>{\n"
        "    html+='<tr>'; columns.forEach(c=>{html+=`<td>${row[c]}</td>`;}); html+='</tr>';\n"
        "  });\n"
        "  html+='</tbody></table>';\n"
        "  document.getElementById('repeatedByTrackResults').innerHTML=html;\n"
        "}\n\n"
        "function updateOITBreakdown(){\n"
        "  console.log('updateOITBreakdown() called');\n"
        "  let trackSel = document.getElementById('oitTrackFilter');\n"
        "  let chosenTracks = Array.from(trackSel.selectedOptions).map(o => o.value);\n"
        "  console.log('Chosen Tracks before check:', chosenTracks);\n"
        "  if (chosenTracks.includes('ALL') || chosenTracks.length === 0) {\n"
        "    chosenTracks = null;\n"
        "  }\n\n"
        "  let seasonSel = document.getElementById('oitSeasonFilter');\n"
        "  let chosenSeasons = Array.from(seasonSel.selectedOptions).map(o => o.value);\n"
        "  console.log('Chosen Seasons before check:', chosenSeasons);\n"
        "  if (chosenSeasons.length === 0) {\n"
        "    chosenSeasons = seasonOrder; // uses the existing \"seasonOrder\" array\n"
        "  }\n\n"
        "  let filtered = oitNonOitData.filter(row => {\n"
        "    let trackMatch = true;\n"
        "    if (chosenTracks) {\n"
        "      trackMatch = chosenTracks.includes(row['Track']);\n"
        "    }\n"
        "    let seasonMatch = chosenSeasons.includes(row['Season']);\n"
        "    return trackMatch && seasonMatch;\n"
        "  });\n"
        "  console.log('Filtered data:', filtered);\n\n"
        "  if (filtered.length === 0) {\n"
        "    document.getElementById('oitBreakdownResults').innerHTML = '<p>No data found.</p>';\n"
        "    return;\n"
        "  }\n\n"
        "  let html = \"<table id='oitTable'><thead><tr>\"\n"
        "           + \"<th>Track</th><th>Season</th><th>OIT vs Non-OIT</th><th>Count</th>\"\n"
        "           + \"</tr></thead><tbody>\";\n\n"
        "  filtered.forEach(row => {\n"
        "    html += \"<tr>\"\n"
        "          + `<td>${row['Track']}</td>`\n"
        "          + `<td>${row['Season']}</td>`\n"
        "          + `<td>${row['OIT vs Non-OIT']}</td>`\n"
        "          + `<td>${row['Count']}</td>`\n"
        "          + \"</tr>\";\n"
        "  });\n\n"
        "  html += \"</tbody></table>\";\n"
        "  document.getElementById('oitBreakdownResults').innerHTML = html;\n"
        "}\n\n"
        "// CHARTS\n"
        "document.addEventListener('DOMContentLoaded',function(){\n"
        "  let ctxSeason=document.getElementById('seasonEnrollmentChart').getContext('2d');\n"
        f"  let so={season_order_json};\n"
        f"  let se={season_enrollment_json};\n"
        f"  let sc={season_completions_json};\n"
        f"  let sd={season_drops_json};\n"
        "  new Chart(ctxSeason,{\n"
        "    type:'line',\n"
        "    data:{\n"
        "      labels:so,\n"
        "      datasets:[\n"
        "        {label:'Total Enrolled',data:se,\n"
        "         backgroundColor:'rgba(54,162,235,0.2)',borderColor:'rgba(54,162,235,1)',\n"
        "         borderWidth:1,fill:false},\n"
        "        {label:'Total Completed',data:sc,\n"
        "         backgroundColor:'rgba(75,192,192,0.2)',borderColor:'rgba(75,192,192,1)',\n"
        "         borderWidth:1,fill:false},\n"
        "        {label:'Total Dropped',data:sd,\n"
        "         backgroundColor:'rgba(255,99,132,0.2)',borderColor:'rgba(255,99,132,1)',\n"
        "         borderWidth:1,fill:false}\n"
        "      ]},\n"
        "    options:{\n"
        "      responsive:true,\n"
        "      plugins:{title:{display:true,text:'Seasonal Enrollment Trends'}},\n"
        "      scales:{y:{beginAtZero:true,ticks:{precision:0}}}\n"
        "    }\n"
        "  });\n\n"
        "  // Pillar distribution\n"
        "  let ctxPillar=document.getElementById('pillarTrendsChart').getContext('2d');\n"
        f"  let pillarData={pillar_datasets_json};\n"
        "  new Chart(ctxPillar,{\n"
        "    type:'bar',\n"
        "    data:{\n"
        "      labels:so,\n"
        "      datasets:pillarData\n"
        "    },\n"
        "    options:{\n"
        "      responsive:true,\n"
        "      plugins:{\n"
        "        title:{display:true,text:'Enrollment Trends by Pillar'},\n"
        "        tooltip:{mode:'index',intersect:false}\n"
        "      },\n"
        "      scales:{x:{stacked:true},y:{stacked:true,beginAtZero:true}}\n"
        "    }\n"
        "  });\n\n"
        "  // Average Completion Rate per Season\n"
        "  let ctxAvg=document.getElementById('avgCompletionChart').getContext('2d');\n"
        f"  let crvals={completion_rate_values_json};\n"
        "  new Chart(ctxAvg,{\n"
        "    type:'line',\n"
        "    data:{\n"
        "      labels:so,\n"
        "      datasets:[{\n"
        "        label:'Average Completion Rate (%)',\n"
        "        data:crvals,\n"
        "        backgroundColor:'rgba(153,102,255,0.2)',\n"
        "        borderColor:'rgba(153,102,255,1)',\n"
        "        borderWidth:1,\n"
        "        fill:false,\n"
        "        tension:0.1\n"
        "      }]\n"
        "    },\n"
        "    options:{\n"
        "      responsive:true,\n"
        "      plugins:{\n"
        "        title:{display:true,text:'Average Completion Rate per Season'},\n"
        "        tooltip:{callbacks:{label:function(ctx){return ctx.parsed.y + '%';}}}\n"
        "      },\n"
        "      scales:{y:{beginAtZero:true,max:100,ticks:{callback:function(val){return val+'%';}}}}\n"
        "    }\n"
        "  });\n\n"
        "  // Distribution of Tracks across Seasons\n"
        "  let ctxDist=document.getElementById('trackDistChart').getContext('2d');\n"
        f"  let distData={distribution_datasets_json};\n"
        "  new Chart(ctxDist,{\n"
        "    type:'bar',\n"
        "    data:{labels:so,datasets:distData},\n"
        "    options:{\n"
        "      responsive:true,\n"
        "      plugins:{title:{display:true,text:'Distribution of Tracks across Seasons'},tooltip:{mode:'index',intersect:false}},\n"
        "      scales:{x:{stacked:false},y:{beginAtZero:true}}\n"
        "    }\n"
        "  });\n"
        "});\n"
        "</script>\n"
        "</div></body></html>\n"
)

    final_html = "".join(html_parts)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Report successfully generated at: {output_html}")


if __name__ == "__main__":
    generate_insights_report(
        input_csv="Data/Sankey_Pull_March.csv",
        output_html="report.html"
    )
