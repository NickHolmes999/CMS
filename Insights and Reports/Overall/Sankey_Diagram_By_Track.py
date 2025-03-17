import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import plotly.subplots as sp
import numpy as np

# Step 1: Load the Data
df = pd.read_csv('Data/WR Awareness Registration_November 5, 2024_07.33.csv', encoding='utf-8')

# Step 2: Data Preprocessing
# Clean column names
df.columns = df.columns.str.strip().str.replace('Â', '').str.replace('\u00a0', '').str.replace('¬†', '')

# Remove rows with missing emails, 'Track Selection', or 'Season Selection'
df = df.dropna(subset=['Email', 'Track Selection', 'Season Selection'])

# Convert relevant columns to strings
df['Email'] = df['Email'].astype(str)
df['Track Selection'] = df['Track Selection'].astype(str)
df['Season Selection'] = df['Season Selection'].astype(str)
df['Verified Complete'] = df['Verified Complete'].astype(str)
df['Dropped'] = df['Dropped'].astype(str)

# Define the season order for chronological sorting
season_order = [
    'Winter 2022', 'Spring 2022', 'Summer 2022', 'Fall 2022',
    'Winter 2023', 'Spring 2023', 'Summer 2023', 'Fall 2023',
    'Winter 2024', 'Spring 2024', 'Summer 2024', 'Fall 2024'
]

# Ensure 'Season Selection' is categorical with the defined order
df['Season Selection'] = pd.Categorical(df['Season Selection'], categories=season_order, ordered=True)

# Function to determine the status per email per season
def get_status(group):
    if (group['Verified Complete'] == 'Yes').any():
        return group[group['Verified Complete'] == 'Yes'].iloc[0]
    elif (group['Dropped'] == 'Yes').any():
        return group[group['Dropped'] == 'Yes'].iloc[0]
    else:
        return group.iloc[0]

# Apply the function to get one entry per email per season
df = df.groupby(['Email', 'Season Selection']).apply(get_status).reset_index(drop=True)

# Sort data by email and season to reflect the chronological order
df = df.sort_values(['Email', 'Season Selection'])

# Anonymize emails to protect privacy
df['Email'] = df['Email'].apply(lambda x: f"User_{abs(hash(x))%10000}")


def create_track_flow(df, initial_track):
    """
    Creates a flow diagram for a specific initial track, only considering courses taken after
    the selected course based on season order
    """
    # Filter for users who took this track
    initial_users = df[df['Track Selection'] == initial_track]['Email'].unique()

    # Get all entries for these users
    track_df = df[df['Email'].isin(initial_users)]

    # Create flow tracking
    flow_counts = defaultdict(int)
    next_course_counts = defaultdict(int)
    completion_status = {initial_track: {'completed': 0, 'dropped': 0}}

    # Process each user's journey
    for user in initial_users:
        user_courses = track_df[track_df['Email'] == user].sort_values('Season Selection')

        # Get all instances where the user took the initial track
        initial_course_instances = user_courses[user_courses['Track Selection'] == initial_track]

        for _, initial_course in initial_course_instances.iterrows():
            # Determine completion status
            status = 'completed' if initial_course['Verified Complete'] == 'Yes' else 'dropped'
            completion_status[initial_track][status] += 1

            # Get the season of the initial course
            initial_season = initial_course['Season Selection']

            # Find courses taken after this instance
            # Convert seasons to indices for comparison
            initial_season_idx = season_order.index(initial_season)

            # Filter for courses taken in later seasons only
            later_courses = user_courses[
                user_courses['Season Selection'].apply(lambda x: season_order.index(x) > initial_season_idx)
            ]

            # If there are any later courses, count the first one
            if not later_courses.empty:
                next_course = later_courses.iloc[0]
                next_track = next_course['Track Selection']

                # Only count if it's a different track
                if next_track != initial_track:
                    flow_counts[(initial_track, next_track, status)] += 1
                    next_course_counts[next_track] += 1

    return completion_status, next_course_counts, flow_counts


def create_track_sankey(initial_track, completion_status, next_course_counts, flow_counts):
    """
    Creates a Sankey diagram for a specific track
    """
    nodes = []
    node_indices = {}

    def get_node_index(name):
        if name not in node_indices:
            node_indices[name] = len(nodes)
            nodes.append(name)
        return node_indices[name]

    # Create nodes
    initial_idx = get_node_index(initial_track)
    completed_idx = get_node_index(f"{initial_track} (Completed)")
    dropped_idx = get_node_index(f"{initial_track} (Dropped)")

    # Create links
    sources = []
    targets = []
    values = []

    # Add completion status links
    sources.extend([initial_idx, initial_idx])
    targets.extend([completed_idx, dropped_idx])
    values.extend([
        completion_status[initial_track]['completed'],
        completion_status[initial_track]['dropped']
    ])

    # Add flow to next courses
    for (source_track, next_track, status), count in flow_counts.items():
        next_idx = get_node_index(next_track)
        status_source = completed_idx if status == 'completed' else dropped_idx

        sources.append(status_source)
        targets.append(next_idx)
        values.append(count)

    # Create summary text of most common next courses
    sorted_next_courses = sorted(next_course_counts.items(), key=lambda x: x[1], reverse=True)
    summary_text = "Most common next courses:<br>"
    for course, count in sorted_next_courses[:3]:  # Top 3 next courses
        summary_text += f"{course}: {count} students<br>"

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=['blue', 'green', 'red'] + ['gray'] * (len(nodes) - 3)
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Flow from {initial_track}",
        font_size=10,
        height=800,
        annotations=[
            dict(
                text=summary_text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.1,
                align="left"
            )
        ]
    )

    return fig

# Get unique tracks
unique_tracks = df['Track Selection'].unique()

# Create a figure for each track
figs = []
for track in unique_tracks:
    completion_status, next_course_counts, flow_counts = create_track_flow(df, track)
    fig = create_track_sankey(track, completion_status, next_course_counts, flow_counts)
    figs.append(fig)

# Create HTML content
html_content = """
<html>
<head>
    <title>Track Flow Analysis</title>
    <style>
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
        }
        .tab button:hover {
            background-color: #ddd;
        }
        .tab button.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }
    </style>
>
<div class="tab">
"""

# Add buttons for each track
for i, track in enumerate(unique_tracks):
    html_content += f'<button class="tablinks" onclick="openTrack(event, \'track{i}\')">{track}</button>'

html_content += '</div>'

# Add div for each track's figure
for i, fig in enumerate(figs):
    html_content += f'<div id="track{i}" class="tabcontent">'
    html_content += fig.to_html(full_html=False)
    html_content += '</div>'

# Add JavaScript to handle tab switching
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

// Open the first tab by default
document.getElementsByClassName("tablinks")[0].click();
</script>
>
"""

# Write to HTML file
with open('track_flow_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Analysis complete! Open track_flow_analysis.html in your web browser to view the results.")
