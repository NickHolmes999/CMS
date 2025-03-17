import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
import re

# Load the Data
df = pd.read_csv('Data\Sankey_Pull_March.csv', encoding='utf-8')

# Clean column names
df.columns = df.columns.str.strip().str.replace('Â', '', regex=False).str.replace('\u00a0', '', regex=False).str.replace('¬†', '', regex=False)

# Remove rows with missing emails or 'Track'
df = df.dropna(subset=['Email', 'Track'])

# Convert relevant columns to strings and strip whitespace
df['Email'] = df['Email'].astype(str).str.strip()
df['Track'] = df['Track'].astype(str).str.strip()
df['Season'] = df['Season'].astype(str).str.strip()
df['Verified Complete'] = df['Verified Complete'].astype(str).str.strip()
df['Dropped'] = df['Dropped'].astype(str).str.strip()
df['Level'] = df['Level'].astype(str).str.strip()
df['Pillar'] = df['Pillar'].astype(str).str.strip()

# Debug: Print unique 'Season' values before standardization
print("Unique 'Season' values before standardization:")
print(df['Season'].unique())

# Function to standardize season names using regular expressions
def standardize_season(season):
    if pd.isnull(season) or season.strip() == '':
        return 'Unknown Season'
    
    season = season.strip()
    season = ' '.join(season.split())  # Remove extra spaces
    season = season.title()  # Standardize capitalization
    # Replace any non-alphanumeric characters with a space
    season = re.sub(r'[^A-Za-z0-9 ]+', ' ', season)
    # Remove extra spaces again
    season = ' '.join(season.split())
    
    # Regular expression patterns to match different formats, including two-digit years and only year
    patterns = [
        r'^(Winter|Spring|Summer|Fall|Autumn)\s*(\d{2,4})$',      # 'Winter 22' or 'Winter 2022'
        r'^(\d{2,4})\s*(Winter|Spring|Summer|Fall|Autumn)$',      # '22 Winter' or '2022 Winter'
        r'^(Win|Spr|Sum|Fal|Aut)\s*(\d{2,4})$',                  # 'Win 22' or 'Win 2022'
        r'^(\d{2,4})\s*(Win|Spr|Sum|Fal|Aut)$',                  # '22 Win' or '2022 Win'
        r'^(Winter|Spring|Summer|Fall|Autumn)(\d{2,4})$',        # 'Winter22' or 'Winter2022'
        r'^(\d{2,4})(Winter|Spring|Summer|Fall|Autumn)$',        # '22Winter' or '2022Winter'
        r'^(Win|Spr|Sum|Fal|Aut)(\d{2,4})$',                     # 'Win22' or 'Win2022'
        r'^(\d{2,4})(Win|Spr|Sum|Fal|Aut)$',                     # '22Win' or '2022Win'
        r'^(\d{2,4})$',                                           # '2022' or '22'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, season)
        if match:
            groups = match.groups()
            if len(groups) == 1 and groups[0]:  # Only year is provided
                year = groups[0]
                if len(year) == 2:
                    year = '20' + year  # Assuming years are 2000s
                # Default to 'Fall' season if only year is given
                return f'Fall {year}'
            elif len(groups) == 2:
                if groups[0].isdigit():
                    year = groups[0]
                    season_name = groups[1]
                else:
                    season_name = groups[0]
                    year = groups[1]
                # Handle two-digit years
                if len(year) == 2:
                    year = '20' + year  # Assuming years are 2000s
                # Handle abbreviations
                season_name = season_name.lower()
                season_mapping = {
                    'winter': 'Winter',
                    'spring': 'Spring',
                    'summer': 'Summer',
                    'fall': 'Fall',
                    'autumn': 'Fall',  # Map 'Autumn' to 'Fall'
                    'win': 'Winter',
                    'spr': 'Spring',
                    'sum': 'Summer',
                    'fal': 'Fall',
                    'aut': 'Fall',
                }
                season_full = season_mapping.get(season_name, 'Unknown Season')
                return f'{season_full} {year}'
    # If no pattern matched, return 'Unknown Season'
    return 'Unknown Season'

# Apply the function to standardize 'Season'
df['Season'] = df['Season'].apply(standardize_season)

# Debug: Print unique seasons after standardization
print("Unique seasons after standardization:")
print(df['Season'].unique())

# Extract years from 'Season'
years_in_data = df['Season'].str.extract(r'(\d{4})')[0].dropna().astype(int).unique()
years_in_data = sorted(years_in_data)

# Manually include missing years if necessary
# For example, if you know that the data should include years from 2021 to 2025
years_needed = range(2021, 2026)
years_in_data = sorted(set(years_in_data).union(set(years_needed)))

# Define the seasons
seasons = ['Winter', 'Spring', 'Summer', 'Fall']

# Create 'season_order' based on the years and seasons present
season_order = [f'{season} {year}' for year in years_in_data for season in seasons]

# Include 'Unknown Season' in season_order if needed
season_order.append('Unknown Season')

# Ensure 'Season' is categorical with the defined order
df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)

# Handle missing 'Season' values (NaN)
df['Season'] = df['Season'].fillna('Unknown Season')

# Check for any 'Unknown Season' values in 'Season' after conversion
unknown_seasons = df[df['Season'] == 'Unknown Season']
if not unknown_seasons.empty:
    print("Warning: The following records have 'Season' values not recognized:")
    print(unknown_seasons[['Email', 'Season']])

# Decide whether to remove or keep records with 'Unknown Season'
# For this analysis, we'll keep them to include all data
# If you prefer to remove them, uncomment the following line:
# df = df[df['Season'] != 'Unknown Season']

# Function to determine the status per email per season
def get_status(group):
    if (group['Verified Complete'].str.lower() == 'yes').any():
        return group[group['Verified Complete'].str.lower() == 'yes'].iloc[0]
    elif (group['Dropped'].str.lower() == 'yes').any():
        return group[group['Dropped'].str.lower() == 'yes'].iloc[0]
    else:
        return group.iloc[0]

# Apply the function to get one entry per email per season
df = df.groupby(['Email', 'Season'], as_index=False).apply(get_status).reset_index(drop=True)

# Sort data by email and season to reflect the chronological order
df = df.sort_values(['Email', 'Season'])

# Anonymize emails to protect privacy
df['Email'] = df['Email'].apply(lambda x: f"User_{abs(hash(x))%10000}")

# Build Full Track Name including Level and Pillar
def build_full_track_name(row):
    class_taken = str(row['Track']).strip()
    level = str(row['Level']).strip()
    pillar = str(row['Pillar']).strip()

    if class_taken.lower() == 'nan' or class_taken == '':
        return None
    if level.lower() == 'nan' or level == '':
        level = ''
    if pillar.lower() == 'nan' or pillar == '':
        pillar = ''

    if level:
        if 'Level' in level:
            class_full_name = f"{class_taken} {level}"
        else:
            class_full_name = f"{class_taken} Level {level}"
    else:
        class_full_name = class_taken

    if pillar:
        class_full_name += f" {pillar}"

    return class_full_name.strip()

df['Full Track Name'] = df.apply(build_full_track_name, axis=1)

# Remove rows where 'Full Track Name' is None or empty (missing class)
df = df[df['Full Track Name'].notna() & (df['Full Track Name'] != '')].reset_index(drop=True)

# Function to calculate statistics for each season for a given track
def calculate_season_statistics(df, track):
    stats = []
    track_df = df[df['Full Track Name'] == track]
    
    # Make sure to use all seasons from the beginning
    all_seasons = pd.Categorical(season_order, categories=season_order, ordered=True)
    
    # Create stats for each season, even if there are no enrollments
    for season in all_seasons:
        season_df = track_df[track_df['Season'] == season]
        
        enrollments = len(season_df)
        completions = (season_df['Verified Complete'].str.lower() == 'yes').sum()
        drops = (season_df['Dropped'].str.lower() == 'yes').sum()
        completion_rate = (completions / enrollments) * 100 if enrollments > 0 else 0
        drop_rate = (drops / enrollments) * 100 if enrollments > 0 else 0
        
        stats.append({
            'Season': season,
            'Enrollments': enrollments,
            'Completions': completions,
            'Drops': drops,
            'Completion Rate (%)': round(completion_rate, 2),
            'Drop Rate (%)': round(drop_rate, 2)
        })

    # Create DataFrame and ensure 'Season' is a column
    season_stats_df = pd.DataFrame(stats)
    
    # Sort by season to maintain chronological order
    season_stats_df['Season'] = pd.Categorical(season_stats_df['Season'], 
                                             categories=season_order, 
                                             ordered=True)
    season_stats_df = season_stats_df.sort_values('Season')
    season_stats_df.reset_index(drop=True, inplace=True)

    return season_stats_df

# Function to create track flow
def create_track_flow(df, initial_track):
    initial_users = df[df['Full Track Name'] == initial_track]['Email'].unique()
    track_df = df[df['Email'].isin(initial_users)]

    flow_counts = defaultdict(int)
    next_course_counts = defaultdict(int)
    completion_status = {initial_track: {'completed': 0, 'dropped': 0}}

    for user in initial_users:
        user_courses = track_df[track_df['Email'] == user].sort_values('Season')

        initial_course_instances = user_courses[user_courses['Full Track Name'] == initial_track]

        for _, initial_course in initial_course_instances.iterrows():
            status = 'completed' if initial_course['Verified Complete'].lower() == 'yes' else 'dropped'

            completion_status[initial_track][status] += 1

            initial_season = initial_course['Season']
            if pd.isnull(initial_season):
                continue  # Skip if 'Season' is missing

            initial_season_idx = season_order.index(initial_season)

            later_courses = user_courses[
                user_courses['Season'].apply(lambda x: season_order.index(x) if x in season_order else -1) > initial_season_idx
            ]

            if not later_courses.empty:
                next_course = later_courses.iloc[0]
                next_track = next_course['Full Track Name']

                if next_track != initial_track:
                    flow_counts[(initial_track, next_track, status)] += 1
                    next_course_counts[next_track] += 1

    return completion_status, next_course_counts, flow_counts

# Create Sankey with stats
def create_track_sankey_with_stats(initial_track, completion_status, next_course_counts, flow_counts):
    nodes = []
    node_indices = {}

    # Function to get the index of a node, adding it if it doesn't exist
    def get_node_index(name):
        if name not in node_indices:
            node_indices[name] = len(nodes)
            nodes.append(name)
        return node_indices[name]

    # Create initial nodes and indices
    initial_idx = get_node_index(initial_track)
    completed_idx = get_node_index(f"{initial_track} (Completed)")
    dropped_idx = get_node_index(f"{initial_track} (Dropped)")

    # Initialize sources, targets, values, and link labels for the Sankey diagram
    sources = [initial_idx, initial_idx]
    targets = [completed_idx, dropped_idx]
    values = [
        completion_status[initial_track]['completed'],
        completion_status[initial_track]['dropped']
    ]
    link_labels = [
        f"{initial_track} to {initial_track} (Completed)",
        f"{initial_track} to {initial_track} (Dropped)"
    ]

    # Add flows to next courses
    for (source_track, next_track, status), count in flow_counts.items():
        next_idx = get_node_index(next_track)
        status_source = completed_idx if status == 'completed' else dropped_idx

        sources.append(status_source)
        targets.append(next_idx)
        values.append(count)
        link_labels.append(f"{nodes[status_source]} to {next_track}")

    # Define node colors
    node_colors = []
    for label in nodes:
        if label == initial_track:
            node_colors.append('blue')
        elif label == f"{initial_track} (Completed)":
            node_colors.append('green')
        elif label == f"{initial_track} (Dropped)":
            node_colors.append('red')
        else:
            node_colors.append('gray')

    # Calculate season statistics
    season_stats_df = calculate_season_statistics(df, initial_track)

    # Rename columns to shorten headers if desired
    season_stats_df.rename(columns={
        'Completion Rate (%)': 'Completion%',
        'Drop Rate (%)': 'Drop%'
    }, inplace=True)

    # Ensure that 'Season' is a column and not an index
    if 'Season' not in season_stats_df.columns:
        season_stats_df.reset_index(inplace=True)

    # Convert 'Season' column to string explicitly
    season_stats_df['Season'] = season_stats_df['Season'].astype(str)

    # Ensure all data is converted to strings before passing to Plotly
    for col in season_stats_df.columns:
        season_stats_df[col] = season_stats_df[col].astype(str)

    # Create the table with adjusted column widths
    column_widths = [100, 80, 80, 80, 100, 100]  # Adjust the widths as needed

    table_fig = go.Figure(data=[go.Table(
        columnwidth=column_widths,
        header=dict(
            values=list(season_stats_df.columns),
            align='center',
            font=dict(size=12),
            height=40
        ),
        cells=dict(
            values=[season_stats_df[col].tolist() for col in season_stats_df.columns],
            align='center',
            height=30
        )
    )])

    # Adjust the layout to accommodate the table
    table_fig.update_layout(
        width=700,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
        )
    )])

    # Update layout for the Sankey diagram
    fig.update_layout(
        title_text=f"Flow and Season Statistics for {initial_track}",
        font_size=10,
        height=800
    )

    return fig, table_fig

# Get unique tracks
unique_tracks = df['Full Track Name'].dropna().unique()
unique_tracks = [track for track in unique_tracks if track != '']

# Debug statement to check unique tracks
print("Unique tracks:")
print(unique_tracks)

figs = []
tables = []
for track in unique_tracks:
    print(f"Processing track: {track}")
    completion_status, next_course_counts, flow_counts = create_track_flow(df, track)
    sankey_fig, table_fig = create_track_sankey_with_stats(track, completion_status, next_course_counts, flow_counts)
    figs.append(sankey_fig)
    tables.append(table_fig)

html_content = """
<html>
<head>
    <title>Track Flow Analysis</title>
    <style>
        .tab { overflow: auto; border: 1px solid #ccc; background-color: #f1f1f1; white-space: nowrap; }
        .tab button { background-color: inherit; display: inline-block; border: none; outline: none; cursor: pointer; padding: 10px 12px; transition: 0.3s; font-size: 12px; }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: #ccc; }
        .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; overflow-x: auto; }
    </style>
</head>
<body>
<div class="tab">
"""

for i, track in enumerate(unique_tracks):
    html_content += f'<button class="tablinks" onclick="openTrack(event, \'track{i}\')">{track}</button>'

html_content += '</div>'

for i, (fig, table) in enumerate(zip(figs, tables)):
    html_content += f'<div id="track{i}" class="tabcontent">'
    html_content += fig.to_html(full_html=False, include_plotlyjs='cdn')
    html_content += table.to_html(full_html=False, include_plotlyjs=False)
    html_content += '</div>'

html_content += """
<script>
function openTrack(evt, trackName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(trackName).style.display = "block";
    evt.currentTarget.className += " active";
}

document.getElementsByClassName("tablinks")[0].click();
</script>
</body>
</html>
"""

with open('track_flow_analysis_with_stats.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Analysis complete! Open track_flow_analysis_with_stats.html in your web browser to view the results.")
