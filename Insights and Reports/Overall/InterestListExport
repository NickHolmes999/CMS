import pandas as pd

def create_interest_list(input_csv="Data/Sankey_Pull_March.csv", output_xlsx="interest_list_only.xlsx"):
    # --- Step 1: Read CSV ---
    df = pd.read_csv(input_csv)

    # --- Step 2: Ensure required columns exist and normalize strings ---
    required_cols = ["Interest List", "Verified Complete", "Track", "Track Selection", "Email", "Email Input"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")
        df[col] = df[col].fillna("").astype(str).str.strip()

    # --- Step 3: Fill in empty Track values using Track Selection ---
    df["Track"] = df["Track"].where(df["Track"] != "", df["Track Selection"])

    # --- Step 4: Fill in empty Email values using Email Input ---
    df["Email"] = df["Email"].where(df["Email"] != "", df["Email Input"])

    # --- Step 5: Remove rows with missing Email or Track ---
    df = df[(df["Email"] != "") & (df["Track"] != "")].copy()

    # --- Step 6: Normalize values to lowercase for matching ---
    df["Interest List"] = df["Interest List"].str.lower()
    df["Verified Complete"] = df["Verified Complete"].str.lower()
    df["Track_lower"] = df["Track"].str.lower()
    df["Email_lower"] = df["Email"].str.lower()

    # --- Step 7: Get ALL interest list entries (regardless of original Track field) ---
    df_interest_all = df[df["Interest List"] == "yes"].copy()
    total_interest_all = len(df_interest_all)

    # --- Step 8: Filter those with non-empty Track and Email ---
    df_interest = df_interest_all[df_interest_all["Track"] != ""].copy()
    total_interest_with_track = len(df_interest)

    # --- Step 9: Extract interest list (email + track) pairs ---
    interest_pairs = df_interest[["Email_lower", "Track_lower", "Email", "Track"]].drop_duplicates()

    # --- Step 10: Build set of completed (email, track) combinations ---
    df_completed = df[df["Verified Complete"] == "yes"]
    completed_set = set(zip(df_completed["Email_lower"], df_completed["Track_lower"]))

    # --- Step 11: Identify which interest list entries already completed the track ---
    interest_pairs["Already_Completed"] = interest_pairs.apply(
        lambda row: (row["Email_lower"], row["Track_lower"]) in completed_set, axis=1
    )

    # --- Step 12: Debug counts ---
    total_completed = interest_pairs["Already_Completed"].sum()
    total_remaining = len(interest_pairs) - total_completed

    # --- Step 13: Filter to only those who have NOT completed the track ---
    df_final = interest_pairs[~interest_pairs["Already_Completed"]].copy()
    df_final = df_final[["Email", "Track"]].sort_values(by=["Email", "Track"])

    # --- Step 14: Write to Excel ---
    df_final.to_excel(output_xlsx, sheet_name="InterestList", index=False, engine="openpyxl")

    # --- Step 15: Print debug summary ---
    print("📊 Summary:")
    print(f" - Total people with Interest List == yes (any Track): {total_interest_all}")
    print(f" - Of those, with usable Track and Email: {total_interest_with_track}")
    print(f" - Of those, already completed that Track: {total_completed}")
    print(f" - Final output (still interested, not completed): {total_remaining}")
    print(f"✅ Created Excel file: {output_xlsx}")


if __name__ == "__main__":
    create_interest_list()
