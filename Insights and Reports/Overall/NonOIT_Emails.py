import pandas as pd

def get_emails_with_empty_group(csv_file_path):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Filter rows where 'Group' is NaN or empty
    condition = df['Group'].isna() | (df['Group'] == "")
    empty_group_df = df[condition]
    
    # Deduplicate the Email column
    deduped_emails = empty_group_df['Email'].drop_duplicates()
    
    # Convert to a Python list
    email_list = deduped_emails.tolist()
    
    return email_list

if __name__ == "__main__":
    # Replace 'people.csv' with the path to your file
    csv_file_path = 'Data/dashboard-export-05-55-pm-2025-01-15.csv'
    
    emails = get_emails_with_empty_group(csv_file_path)
    
    # Print the list of emails
    print(emails)
