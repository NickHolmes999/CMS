import streamlit as st
import pandas as pd

def main():
    st.title("CSV Filter & Query App")

    # 1. Load the CSV data
    # Replace 'mydata.csv' with your actual filename or path to your CSV
    df = pd.read_csv('Data/WR Sankey  copy.csv', dtype=str).fillna('')

    # 2. Define the columns you want to filter
    filter_columns = [
        "CMS Center Selection",
        "OIT Group",
        "Dropped",
        "Verified Complete",
        "Season Selection",
        "Track Selection",
        "Session Selection",
        "Email",
        "Pillar",
        "Offering",
        "Motivation",
        "Motivation_9_TEXT",
        "Level",
        "Special Interests",
        "CMS Group",
        "CMS Division",
        "BA Familiarity",
        "Benefits of Agile",
        "Applications of Agile",
        "Topic Interest",
        "Track Familiarity",
        "Benefits of Track",
        "Track Application",
        "Track Motivation",
        "Track Motivation_9_TEXT"
    ]

    # We will store user selections in a dictionary {column_name: [selected_values]}
    user_filters = {}

    # 3. Create multiselect widgets for each column
    for col in filter_columns:
        # Get unique values
        unique_values = sorted(set(df[col]))
        # Create an "All" option
        all_option_label = "All"
        filter_options = [all_option_label] + unique_values

        selected = st.multiselect(
            f"Select {col} (choose multiple or 'All')",
            options=filter_options,
            default=[all_option_label]
        )

        # If "All" is in the selection, use all unique values
        if all_option_label in selected:
            user_filters[col] = unique_values
        else:
            user_filters[col] = selected

    # 4. Filter the dataframe
    filtered_df = df.copy()
    for col in filter_columns:
        chosen_values = user_filters[col]
        filtered_df = filtered_df[filtered_df[col].isin(chosen_values)]

    # 5. Display the results
    st.subheader("Filtered Results")
    st.write(f"Number of matching records: {len(filtered_df)}")
    st.dataframe(filtered_df.reset_index(drop=True))

    # 6. Add a button to save the filtered data to an HTML file
    if st.button("Save to HTML"):
        # Convert dataframe to HTML
        filtered_html = filtered_df.to_html(index=False)
        # Write to local HTML file
        with open("filtered_data.html", "w", encoding="utf-8") as f:
            f.write(filtered_html)
        st.success("Filtered data saved to 'filtered_data.html'")

if __name__ == "__main__":
    main()
