import pandas as pd

def create_special_lists(input_csv="Data/Sankey_Pull_March.csv", output_xlsx="special_lists.xlsx"):
    """
    Reads the CSV, applies logic for:
      1) InterestList (fallback Track/Email, filters by verified completion)
      2) MailingList (must have 'All upcoming offerings' in mailing list interests AND no completion)
      3) NonOITCompleted
      4) Combined (unique union of all lists)
    Writes all lists to Excel as separate sheets.
    """

    # STEP 1: Read CSV
    df = pd.read_csv(input_csv)

    # STEP 2: Clean and normalize key columns
    required_cols = [
        "Interest List", "Mailing list interests", "Verified Complete",
        "CMS Center", "Track", "Track Selection", "Email", "Email Input"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")
        df[col] = df[col].fillna("").astype(str).str.strip()

    # STEP 3: Fallback for missing Track and Email values
    df["Track"] = df["Track"].where(df["Track"] != "", df["Track Selection"])
    df["Email"] = df["Email"].where(df["Email"] != "", df["Email Input"])

    # STEP 4: Normalize key fields for filtering
    df["Interest List"] = df["Interest List"].str.lower()
    df["Verified Complete"] = df["Verified Complete"].str.lower()
    df["CMS Center"] = df["CMS Center"].str.lower()
    df["Mailing list interests"] = df["Mailing list interests"].str.strip().str.lower()
    df["Track_lower"] = df["Track"].str.lower()
    df["Email_lower"] = df["Email"].str.lower()

    # STEP 5: Print unique cleaned values to verify
    print("\n📋 Unique values in 'Mailing list interests' AFTER CLEANING:")
    print(df["Mailing list interests"].unique())

    # =====================================================================
    # A) INTEREST LIST
    # =====================================================================
    df_interest_all = df[df["Interest List"] == "yes"].copy()
    df_interest_all = df_interest_all[df_interest_all["Track"] != ""].copy()

    interest_pairs = df_interest_all[["Email_lower", "Track_lower", "Email", "Track"]].drop_duplicates()

    df_completed = df[df["Verified Complete"] == "yes"]
    completed_set = set(zip(df_completed["Email_lower"], df_completed["Track_lower"]))

    interest_pairs["Already_Completed"] = interest_pairs.apply(
        lambda row: (row["Email_lower"], row["Track_lower"]) in completed_set, axis=1
    )

    interest_list_final = interest_pairs[~interest_pairs["Already_Completed"]].copy()
    interest_list_final = interest_list_final[["Email"]].drop_duplicates().sort_values(by="Email")

    # =====================================================================
    # B) MAILING LIST — Only with 'all upcoming offerings' AND no completion
    # =====================================================================
    mailing_interest_df = df[df["Mailing list interests"] == "all upcoming offerings"].copy()

    # Get emails that match this criteria
    mailing_list_emails = mailing_interest_df["Email"].unique()

    # Check each email: include only if they have NO Verified Complete == 'yes'
    def has_never_completed(email):
        return not (df[(df["Email"] == email) & (df["Verified Complete"] == "yes")].any().any())

    filtered_mailing_emails = [email for email in mailing_list_emails if has_never_completed(email)]

    mailing_list_final = pd.DataFrame({"Email": sorted(filtered_mailing_emails)})

    print(f"\n🔍 Found {len(mailing_list_final)} mailing list emails (all upcoming offerings + never completed).")

    # =====================================================================
    # C) NON-OIT COMPLETED
    # =====================================================================
    def non_oit_completed(email_df):
        for _, row in email_df.iterrows():
            if row["Verified Complete"] == "yes" and "oit" not in row["CMS Center"]:
                return True
        return False

    non_oit_grouped = df.groupby("Email", group_keys=False).apply(non_oit_completed)
    non_oit_completed_emails = non_oit_grouped[non_oit_grouped == True].index

    # =====================================================================
    # D) COMBINED LIST
    # =====================================================================
    all_interest = set(interest_list_final["Email"])
    all_mailing = set(mailing_list_final["Email"])
    all_non_oit = set(non_oit_completed_emails)
    combined_set = all_interest.union(all_mailing).union(all_non_oit)

    print(f"🔍 Overlap: InterestList ∩ NonOITCompleted: {len(all_interest.intersection(all_non_oit))}")
    print(f"🔍 Overlap: InterestList ∩ MailingList: {len(all_interest.intersection(all_mailing))}")
    print(f"🔍 Overlap: MailingList ∩ NonOITCompleted: {len(all_mailing.intersection(all_non_oit))}")

    # =====================================================================
    # EXPORT TO EXCEL
    # =====================================================================
    df_interest_list = pd.DataFrame({"Email": sorted(all_interest)})
    df_mailing_list = pd.DataFrame({"Email": sorted(all_mailing)})
    df_non_oit_list = pd.DataFrame({"Email": sorted(all_non_oit)})
    df_combined = pd.DataFrame({"Email": sorted(combined_set)})

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df_interest_list.to_excel(writer, sheet_name="InterestList", index=False)
        df_mailing_list.to_excel(writer, sheet_name="MailingList", index=False)
        df_non_oit_list.to_excel(writer, sheet_name="NonOITCompleted", index=False)
        df_combined.to_excel(writer, sheet_name="Combined", index=False)

    print(f"\n✅ Created Excel file '{output_xlsx}' with 4 sheets:")
    print(f" - InterestList: {len(df_interest_list)}")
    print(f" - MailingList: {len(df_mailing_list)}")
    print(f" - NonOITCompleted: {len(df_non_oit_list)}")
    print(f" - Combined (unique union): {len(df_combined)}")

if __name__ == "__main__":
    create_special_lists(
        input_csv="Data/Sankey_Pull_March.csv",
        output_xlsx="special_lists.xlsx"
    )
