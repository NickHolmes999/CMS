#!/usr/bin/env python3
import pandas as pd

def main():
    # ----------------------------------------------------------------
    # 1. READ DATA
    # ----------------------------------------------------------------
    df = pd.read_csv("Data/WR Sankey  copy.csv")

    # Clean up Verified Complete column to standardize yes/no checks
    df["Verified Complete"] = df["Verified Complete"].fillna("").str.lower()

    # ----------------------------------------------------------------
    # 2. FILTER TO COMPLETIONS ONLY (Verified Complete == "yes")
    # ----------------------------------------------------------------
    df_complete = df[df["Verified Complete"] == "yes"].copy()

    # ----------------------------------------------------------------
    # 3. CLASSIFY ROWS AS OIT OR NON-OIT
    #    - If OIT Group column is non-empty => OIT
    #    - Otherwise => Non-OIT
    # ----------------------------------------------------------------
    def classify_oit(text):
        if pd.isna(text) or str(text).strip() == "":
            return "Non-OIT"
        return "OIT"

    df_complete["OIT_Status"] = df_complete["OIT Group"].apply(classify_oit)

    # ----------------------------------------------------------------
    # 4. A) TOTAL COMPLETIONS (rows) & OIT vs NON-OIT
    # ----------------------------------------------------------------
    total_completions = len(df_complete)
    oit_completions = len(df_complete[df_complete["OIT_Status"] == "OIT"])
    non_oit_completions = len(df_complete[df_complete["OIT_Status"] == "Non-OIT"])

    pct_oit_completions = (oit_completions / total_completions * 100) if total_completions else 0
    pct_non_oit_completions = (non_oit_completions / total_completions * 100) if total_completions else 0

    # ----------------------------------------------------------------
    # 4. B) UNIQUE LEARNERS (distinct Emails) & OIT vs. Non-OIT
    #
    # To avoid double-counting an individual as both OIT and Non-OIT,
    # we'll classify each unique learner based on *any* OIT completions.
    # ----------------------------------------------------------------

    # Group by email, find if they've *ever* had an OIT row
    email_oit_flag = df_complete.groupby("Email")["OIT_Status"] \
                                .apply(lambda statuses: any(s == "OIT" for s in statuses))

    # Build a DataFrame of unique learners, labeling them OIT or Non-OIT
    unique_learners_df = pd.DataFrame({"Email": email_oit_flag.index,
                                       "IsOITLearner": email_oit_flag.values})

    # Count how many distinct learners total
    total_unique_learners = len(unique_learners_df)

    # OIT learners: those with IsOITLearner == True
    oit_unique_learners = unique_learners_df["IsOITLearner"].sum()  # sum of bools = count of True
    non_oit_unique_learners = total_unique_learners - oit_unique_learners

    pct_oit_unique = (oit_unique_learners / total_unique_learners * 100) if total_unique_learners else 0
    pct_non_oit_unique = (non_oit_unique_learners / total_unique_learners * 100) if total_unique_learners else 0

    # ----------------------------------------------------------------
    # 4. C) REPEAT LEARNER COMPLETIONS (original approach)
    #
    # "Repeat" means an Email that appears more than once among completions.
    # -> This counts *all rows* from repeat learners (the 'duplicate allowed' approach)
    # ----------------------------------------------------------------
    # 1) Count how many completions each person has:
    completions_per_email = df_complete.groupby("Email")["Email"].transform("size")

    # 2) Mark repeat vs. non-repeat
    df_complete["IsRepeatLearner"] = completions_per_email > 1

    # 3) Filter only repeat learners
    df_repeats = df_complete[df_complete["IsRepeatLearner"]]

    # 4) Number of total *rows* from repeat learners
    total_repeat_completions_dup = len(df_repeats)

    # Split by OIT vs. Non-OIT in these repeated rows
    repeat_oit_completions_dup = len(df_repeats[df_repeats["OIT_Status"] == "OIT"])
    repeat_non_oit_completions_dup = len(df_repeats[df_repeats["OIT_Status"] == "Non-OIT"])

    pct_repeat_oit_dup = (repeat_oit_completions_dup / total_repeat_completions_dup * 100) \
                            if total_repeat_completions_dup else 0
    pct_repeat_non_oit_dup = (repeat_non_oit_completions_dup / total_repeat_completions_dup * 100) \
                                if total_repeat_completions_dup else 0

    # ----------------------------------------------------------------
    # 5. SPLITTING THE 1422 COMPLETIONS
    #    into Single-Completion vs. Repeat-Learner (BUT counting each
    #    repeat learner's multiple completions as multiple rows).
    #
    #    We want a new metric where we can count:
    #     - Single-Completion completions  (these come from learners who have exactly 1 completion row)
    #     - Repeat-Learner completions     (these come from learners who have 2+ completion rows)
    #
    #    And these two numbers add up to total_completions (which is 1422).
    # ----------------------------------------------------------------
    # Identify Single-Completion vs. Repeat-Learner at the row level
    completions_count_by_email = df_complete.groupby("Email")["Email"].transform("size")
    df_complete["IsSingleCompletion"] = (completions_count_by_email == 1)

    # Single-Completion completions (rows)
    single_completion_df = df_complete[df_complete["IsSingleCompletion"]]
    single_completion_count = len(single_completion_df)

    # Repeat-Learner completions (rows)
    repeat_completion_df = df_complete[~df_complete["IsSingleCompletion"]]
    repeat_completion_count = len(repeat_completion_df)

    # Check sum => single_completion_count + repeat_completion_count = total_completions
    # (Should match 1422 in your example data.)

    # 5.A) OIT vs. Non-OIT breakdown for Single-Completion
    single_oit_count = len(single_completion_df[single_completion_df["OIT_Status"] == "OIT"])
    single_non_oit_count = len(single_completion_df[single_completion_df["OIT_Status"] == "Non-OIT"])

    # 5.B) OIT vs. Non-OIT breakdown for Repeat-Learner completions
    repeat_oit_count = len(repeat_completion_df[repeat_completion_df["OIT_Status"] == "OIT"])
    repeat_non_oit_count = len(repeat_completion_df[repeat_completion_df["OIT_Status"] == "Non-OIT"])

    # 5.C) Percentages
    # For single-completion portion
    pct_single_oit = (single_oit_count / single_completion_count * 100) if single_completion_count else 0
    pct_single_non_oit = (single_non_oit_count / single_completion_count * 100) if single_completion_count else 0

    # For repeat-learner portion
    pct_repeat_oit = (repeat_oit_count / repeat_completion_count * 100) if repeat_completion_count else 0
    pct_repeat_non_oit = (repeat_non_oit_count / repeat_completion_count * 100) if repeat_completion_count else 0

    # ----------------------------------------------------------------
    # 6. PRINT RESULTS
    # ----------------------------------------------------------------
    print("=====================================")
    print("       COMPLETION ANALYSIS (NEW)     ")
    print("=====================================")

    # A) TOTAL COMPLETIONS
    print("A) TOTAL COMPLETIONS (rows) where Verified Complete = 'Yes'")
    print("-----------------------------------------------------------")
    print(f"Total Completions:       {total_completions}")
    print(f"OIT Completions:         {oit_completions} ({pct_oit_completions:.2f}%)")
    print(f"Non-OIT Completions:     {non_oit_completions} ({pct_non_oit_completions:.2f}%)")

    # B) UNIQUE LEARNER COMPLETIONS
    print("\nB) UNIQUE LEARNERS (deduplicated by Email, classified by any OIT usage)")
    print("---------------------------------------------------------------------")
    print(f"Total Unique Learners:   {total_unique_learners}")
    print(f"OIT Learners:            {oit_unique_learners} ({pct_oit_unique:.2f}%)")
    print(f"Non-OIT Learners:        {non_oit_unique_learners} ({pct_non_oit_unique:.2f}%)")
    print("Note: A learner is considered OIT if they have *any* OIT completions; otherwise they are Non-OIT.")

    # C) REPEAT LEARNER COMPLETIONS (WITH DUPLICATION)
    print("\nC) REPEAT LEARNER COMPLETIONS (Rows, 'Duplication Allowed')")
    print("-----------------------------------------------------------")
    print("These numbers count every completion row from any repeat learner.")
    print(f"Total Repeat Completions (rows):   {total_repeat_completions_dup}")
    print(f" - OIT Repeat Completions:         {repeat_oit_completions_dup} ({pct_repeat_oit_dup:.2f}%)")
    print(f" - Non-OIT Repeat Completions:     {repeat_non_oit_completions_dup} ({pct_repeat_non_oit_dup:.2f}%)")

    print("""
Explanation:
  - If someone has 5 completions, all 5 are counted here.
  - This can be useful to see how many total repeated enrollments are OIT vs Non-OIT.
  - However, it doesn't align with single-completion + repeat-completion summing up to total,
    because those 5 completions come from 1 learner, heavily skewing counts.
""")

    # D) SINGLE-COMPLETION vs. REPEAT-LEARNER SPLIT (DE-DUPED at learner level, but
    #    STILL counting all completions for repeat learners).
    print("\nD) SPLITTING THE TOTAL COMPLETIONS INTO SINGLE vs. REPEAT-LEARNER COMPLETIONS")
    print("------------------------------------------------------------------------------")
    print(f" - Single-Completion (rows)        = {single_completion_count}")
    print(f"   * OIT:                          {single_oit_count} ({pct_single_oit:.2f}%)")
    print(f"   * Non-OIT:                      {single_non_oit_count} ({pct_single_non_oit:.2f}%)\n")

    print(f" - Repeat-Learner Completions      = {repeat_completion_count}")
    print(f"   * OIT:                          {repeat_oit_count} ({pct_repeat_oit:.2f}%)")
    print(f"   * Non-OIT:                      {repeat_non_oit_count} ({pct_repeat_non_oit:.2f}%)\n")

    print("Check Sum => Single-Completion + Repeat-Learner Completions = Total Completions:")
    print(f"             {single_completion_count} + {repeat_completion_count} = {total_completions}")

    print("""
Explanation:
  - "Single-Completion" means the learner has exactly 1 completion row overall.
  - "Repeat-Learner Completions" means the *total number of completions* from learners who have 2+ completions.
  - Summing these two categories equals the total of all completions (e.g., 1422).
  - This 'Single vs. Repeat' breakdown is *different* from the 'C) Repeat Learner Completions (Duplication Allowed)'
    if you're comparing row counts for repeated learners alone, because section C includes *only* repeated learners,
    while this breakdown considers the entire population (single and repeat) in two buckets.
""")

if __name__ == "__main__":
    main()
