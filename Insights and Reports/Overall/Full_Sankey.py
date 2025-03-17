import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

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
    else:
        return group.iloc[0]

# Apply the function to get one entry per email per season
df = df.groupby(['Email', 'Season Selection']).apply(get_status).reset_index(drop=True)

# Sort data by email and season to reflect the chronological order
df = df.sort_values(['Email', 'Season Selection'])

# Anonymize emails to protect privacy
df['Email'] = df['Email'].apply(lambda x: f"User_{abs(hash(x))%10000}")

# Step 3: Build Nodes and Links for the Sankey Diagram

nodes = []
node_indices = {}
links = defaultdict(lambda: {'source': None, 'target': None, 'value': 0, 'label': '', 'hoverinfo': ''})

# For tracking learning paths
learning_paths_6_steps = {}
learning_paths_5_steps = {}
learning_paths_long = {}
learning_paths_medium = {}
learning_paths_short = {}
# For tracking statistics
user_journeys = {}  # Dictionary to store each user's journey
step_counts = defaultdict(int)
completed_counts = defaultdict(int)

# Helper function to get node index
def get_node(label):
    if label not in node_indices:
        node_indices[label] = len(nodes)
        nodes.append(label)
    return node_indices[label]

for email, group in df.groupby('Email'):
    previous_node = None
    group = group.sort_values('Season Selection')  # Ensure chronological order

    # For building the learning path of this user
    path = []
    step = 1  # To differentiate subsequent classes
    journey = []  # To store the user's journey for statistics

    for idx, row in group.iterrows():
        # Class Node with Step Numbers for visualization
        class_taken = row['Track Selection']
        if pd.isnull(class_taken) or class_taken.strip() == '':
            continue  # Skip if 'Track Selection' is missing or empty

        class_node_label = f"{class_taken} {step}"
        class_node = get_node(class_node_label)

        # Completion Status Node
        status = 'Completed' if row['Verified Complete'] == 'Yes' else 'Dropped'
        status_node_label = f"{class_node_label} - {status}"
        status_node = get_node(status_node_label)

        # Build the learning path with class name and status
        path_entry = f"{class_taken} - {status}"
        path.append(path_entry)

        # Append to journey
        journey.append({'class': class_taken, 'status': status})

        # Link from Class to Completion Status
        link_key = (class_node, status_node)
        links[link_key]['source'] = class_node
        links[link_key]['target'] = status_node
        links[link_key]['value'] += 1
        links[link_key]['label'] = f"{class_node_label} to {status_node_label}"
        links[link_key]['hoverinfo'] = f"From {class_node_label} to {status_node_label}: {links[link_key]['value']} learners"

        # Link from Previous Status Node to Current Class Node
        if previous_node is not None:
            link_key = (previous_node, class_node)
            links[link_key]['source'] = previous_node
            links[link_key]['target'] = class_node
            links[link_key]['value'] += 1
            links[link_key]['label'] = f"{nodes[previous_node]} to {class_node_label}"
            links[link_key]['hoverinfo'] = f"From {nodes[previous_node]} to {class_node_label}: {links[link_key]['value']} learners"

        previous_node = status_node  # Update previous node for the next iteration
        step += 1  # Increment step for next class

    # Store the user's journey
    user_journeys[email] = journey

    # Update statistics
    total_steps = len(journey)
    step_counts[total_steps] += 1

    # Count how many classes the user completed
    completed_classes = sum(1 for entry in journey if entry['status'] == 'Completed')
    completed_counts[completed_classes] += 1

    # Convert the path list to a tuple (so it can be used as a dict key)
    path_tuple = tuple(path)
    if len(path_tuple) == 6:  # Paths with exactly 6 events
        if path_tuple in learning_paths_6_steps:
            learning_paths_6_steps[path_tuple] += 1
        else:
            learning_paths_6_steps[path_tuple] = 1
    elif len(path_tuple) == 5:  # Paths with exactly 5 events
        if path_tuple in learning_paths_5_steps:
            learning_paths_5_steps[path_tuple] += 1
        else:
            learning_paths_5_steps[path_tuple] = 1
    elif len(path_tuple) >= 4:  # Paths with at least 4 events
        if path_tuple in learning_paths_long:
            learning_paths_long[path_tuple] += 1
        else:
            learning_paths_long[path_tuple] = 1
    elif len(path_tuple) == 3:  # Paths with exactly 3 events
        if path_tuple in learning_paths_medium:
            learning_paths_medium[path_tuple] += 1
        else:
            learning_paths_medium[path_tuple] = 1
    elif len(path_tuple) == 2:  # Paths with exactly 2 events
        if path_tuple in learning_paths_short:
            learning_paths_short[path_tuple] += 1
        else:
            learning_paths_short[path_tuple] = 1

# Prepare the lists for the Sankey diagram
source_indices = []
target_indices = []
values = []
link_labels = []
hover_texts = []

for link in links.values():
    source_indices.append(link['source'])
    target_indices.append(link['target'])
    values.append(link['value'])
    link_labels.append(link['label'])
    hover_texts.append(link['hoverinfo'])

# Step 4: Compute Top 5 Most Common Learning Paths

# Sort the learning paths by frequency
sorted_paths_6_steps = sorted(learning_paths_6_steps.items(), key=lambda x: x[1], reverse=True)
sorted_paths_5_steps = sorted(learning_paths_5_steps.items(), key=lambda x: x[1], reverse=True)
sorted_paths_long = sorted(learning_paths_long.items(), key=lambda x: x[1], reverse=True)
sorted_paths_medium = sorted(learning_paths_medium.items(), key=lambda x: x[1], reverse=True)
sorted_paths_short = sorted(learning_paths_short.items(), key=lambda x: x[1], reverse=True)

# Get the top 5 paths
top_n = 5
top_paths_6_steps = sorted_paths_6_steps[:top_n]
top_paths_5_steps = sorted_paths_5_steps[:top_n]
top_paths_long = sorted_paths_long[:top_n]
top_paths_medium = sorted_paths_medium[:top_n]
top_paths_short = sorted_paths_short[:top_n]

# Prepare the text to display
top_paths_6_steps_text = "Top 5 Most Common Learning Paths (Exactly 6 Steps):\n"
for idx, (path, count) in enumerate(top_paths_6_steps, 1):
    path_str = " -> ".join(path)
    top_paths_6_steps_text += f"\n{idx}. {path_str} (Count: {count})"

top_paths_5_steps_text = "\n\nTop 5 Most Common Learning Paths (Exactly 5 Steps):\n"
for idx, (path, count) in enumerate(top_paths_5_steps, 1):
    path_str = " -> ".join(path)
    top_paths_5_steps_text += f"\n{idx}. {path_str} (Count: {count})"

top_paths_long_text = "\n\nTop 5 Most Common Learning Paths (At Least 4 Steps):\n"
for idx, (path, count) in enumerate(top_paths_long, 1):
    path_str = " -> ".join(path)
    top_paths_long_text += f"\n{idx}. {path_str} (Count: {count})"

top_paths_medium_text = "\n\nTop 5 Most Common Learning Paths (Exactly 3 Steps):\n"
for idx, (path, count) in enumerate(top_paths_medium, 1):
    path_str = " -> ".join(path)
    top_paths_medium_text += f"\n{idx}. {path_str} (Count: {count})"

top_paths_short_text = "\n\nTop 5 Most Common Learning Paths (Exactly 2 Steps):\n"
for idx, (path, count) in enumerate(top_paths_short, 1):
    path_str = " -> ".join(path)
    top_paths_short_text += f"\n{idx}. {path_str} (Count: {count})"

# Step 4b: Compute Additional Statistics

# Total number of users
total_users = len(user_journeys)

# How many people made it to specific steps in their journey
steps_reached_counts = defaultdict(int)
for journey in user_journeys.values():
    steps_reached = len(journey)
    steps_reached_counts[steps_reached] += 1

# How many people completed one class
completed_one_class = completed_counts.get(1, 0)

# How many people completed two classes
completed_two_classes = completed_counts.get(2, 0)

# How many people completed first class and signed up for another
completed_first_signed_another = sum(1 for journey in user_journeys.values()
                                     if len(journey) > 1 and journey[0]['status'] == 'Completed')

# How many people dropped after first class and never signed up for another
dropped_first_never_returned = sum(1 for journey in user_journeys.values()
                                   if len(journey) == 1 and journey[0]['status'] == 'Dropped')

# How many people dropped first class and then signed up for another
dropped_first_signed_another = sum(1 for journey in user_journeys.values()
                                   if len(journey) > 1 and journey[0]['status'] == 'Dropped')

# How many people have completed 3 classes, 4 classes, 5 classes, 6 classes, 7 classes
completed_n_classes = {n: sum(1 for journey in user_journeys.values()
                              if sum(1 for entry in journey if entry['status'] == 'Completed') == n)
                       for n in range(3, 8)}

# Calculate the average number of classes completed by users
total_completed_classes = sum(completed_counts[n] * n for n in completed_counts)
average_completed_classes = total_completed_classes / total_users if total_users > 0 else 0

# Calculate the most popular class
class_frequencies = defaultdict(int)
for journey in user_journeys.values():
    for entry in journey:
        class_frequencies[entry['class']] += 1
most_popular_class = max(class_frequencies, key=class_frequencies.get)
most_popular_class_count = class_frequencies[most_popular_class]

# Additional statistics computation

# Initialize dictionaries to store counts
completion_counts = defaultdict(lambda: {'completed': 0, 'total': 0})
followup_counts = defaultdict(lambda: {'followed_up': 0, 'total': 0})

# Iterate over each user's journey to fill in the counts
for journey in user_journeys.values():
    for i, entry in enumerate(journey):
        class_name = entry['class']
        status = entry['status']

        # Update completion counts
        completion_counts[class_name]['total'] += 1
        if status == 'Completed':
            completion_counts[class_name]['completed'] += 1

        # Update follow-up counts
        followup_counts[class_name]['total'] += 1
        if i < len(journey) - 1:
            followup_counts[class_name]['followed_up'] += 1

# Calculate completion rates
completion_rates = {class_name: (counts['completed'] / counts['total']) if counts['total'] > 0 else 0
                    for class_name, counts in completion_counts.items()}

# Calculate follow-up rates
followup_rates = {class_name: (counts['followed_up'] / counts['total']) if counts['total'] > 0 else 0
                  for class_name, counts in followup_counts.items()}

# Find class with highest and lowest completion rates
highest_completion_class = max(completion_rates, key=completion_rates.get)
lowest_completion_class = min(completion_rates, key=completion_rates.get)

# Find class with highest and lowest follow-up rates
highest_followup_class = max(followup_rates, key=followup_rates.get)
lowest_followup_class = min(followup_rates, key=followup_rates.get)

# Find the least taken class
least_taken_class = min(class_frequencies, key=class_frequencies.get)

# Prepare the reorganized additional statistics text
stats_text = "Additional Statistics:\n"
stats_text += f"\nTotal number of users: {total_users}\n"
stats_text += "\nNumber of users who made it to specific steps in their journey:"
for step_num in sorted(steps_reached_counts.keys()):
    count = steps_reached_counts[step_num]
    stats_text += f"\n  Reached Step {step_num}: {count} users"

stats_text += "\n\nNumber of people who completed N classes:"
stats_text += f"\n  Completed 1 class: {completed_one_class} users"
stats_text += f"\n  Completed 2 classes: {completed_two_classes} users"
for n in range(3, 8):
    count = completed_n_classes.get(n, 0)
    stats_text += f"\n  Completed {n} classes: {count} users"

stats_text += f"\n\nAverage number of classes completed by users: {average_completed_classes:.2f}"
stats_text += f"\nMost popular class: {most_popular_class} (Taken {most_popular_class_count} times)"

stats_text += f"\n\nNumber of people who completed their first class and signed up for another: {completed_first_signed_another}"
stats_text += f"\nNumber of people who dropped after the first class and never signed up for another: {dropped_first_never_returned}"
stats_text += f"\nNumber of people who dropped the first class and then signed up for another: {dropped_first_signed_another}"

# Prepare the additional class-specific statistics text
additional_stats_text = "\n\nAdditional Class Statistics:\n"
additional_stats_text += f"\nClass with the highest completion rate: {highest_completion_class} ({completion_rates[highest_completion_class]:.2%})"
additional_stats_text += f"\nClass with the lowest completion rate: {lowest_completion_class} ({completion_rates[lowest_completion_class]:.2%})"
additional_stats_text += f"\nClass with the highest follow-up rate: {highest_followup_class} ({followup_rates[highest_followup_class]:.2%})"
additional_stats_text += f"\nClass with the lowest follow-up rate: {lowest_followup_class} ({followup_rates[lowest_followup_class]:.2%})"
additional_stats_text += f"\nClass taken the least: {least_taken_class} (Taken {class_frequencies[least_taken_class]} times)"

# Combine the top paths text, reorganized statistics, and additional class statistics
full_text = (top_paths_6_steps_text + top_paths_5_steps_text + top_paths_long_text +
             top_paths_medium_text + top_paths_short_text + "\n\n" + stats_text + additional_stats_text)

# Prepare the text with line breaks for annotation
formatted_text = full_text.replace('\n', '<br>')

# Step 5: Create the Sankey Diagram

# Assign colors to nodes based on classes
class_colors = {
    'Cloud': 'blue',
    'Cyber-Hygiene: Essentials': 'yellow',
    'Data Science': 'orange',
    'Human-Centered Design': 'purple',
    'Artificial Intelligence and Machine Learning': 'pink',
    'Cybersecurity': 'yellow',
    'AI/ML': 'lightpink',
    'CMS 101': 'cyan',
    'Product Management': 'red',
    'Leadership': 'brown',
    'Completed': 'lightgreen',  # For status nodes
    'Dropped': 'lightcoral'
}

# Assign colors to nodes
node_colors = []
for label in nodes:
    # Ensure label is a string
    label = str(label)
    # Check if the label contains a class name or status
    class_found = False
    for class_name, color in class_colors.items():
        if class_name in label:
            node_colors.append(color)
            class_found = True
            break
    if not class_found:
        node_colors.append('grey')  # Default color if no class matches

# Assign positions to nodes to spread them out horizontally
max_step = 0
node_steps = {}

def get_step_number(label):
    # For status nodes
    if ' - ' in label:
        # Split the label by ' - '
        class_part, status_part = label.split(' - ', 1)
    else:
        class_part = label

    # Now, split the class_part by space from the end to find the step number
    tokens = class_part.strip().split(' ')
    # The step number is the last token if it's a number
    step_str = tokens[-1]
    try:
        step = int(step_str)
    except ValueError:
        # If we can't parse an integer, default to 1
        step = 1
    return step

for label in nodes:
    step = get_step_number(label)
    if ' - ' in label:
        # It's a status node
        step = step * 2
    else:
        # It's a class node
        step = step * 2 - 1
    node_steps[label] = step
    if step > max_step:
        max_step = step

# Calculate x positions based on steps
node_x = [0] * len(nodes)
node_y = [0] * len(nodes)

for label, index in node_indices.items():
    step = node_steps[label]
    x_pos = step / (max_step + 1)  # Normalize x position between 0 and 1
    node_x[index] = x_pos

# Group node indices by step to distribute them vertically
step_nodes = defaultdict(list)
for label, index in node_indices.items():
    step = node_steps[label]
    step_nodes[step].append(index)

# Distribute nodes vertically per step
for step, indices in step_nodes.items():
    num_nodes = len(indices)
    if num_nodes == 1:
        # If there's only one node at this step, position it in the middle
        node_y[indices[0]] = 0.5
    else:
        # Evenly space nodes between 0 and 1
        for i, idx in enumerate(indices):
            node_y[idx] = i / (num_nodes - 1)
            # Slightly adjust positions to avoid nodes being at exact 0 or 1
            node_y[idx] = 0.05 + 0.9 * node_y[idx]

# Create the figure with adjusted node spacing and positions
fig = go.Figure(data=[go.Sankey(
    arrangement="freeform",  # Allow manual positioning
    node=dict(
        pad=40,  # Increased padding between nodes
        thickness=15,  # Decreased thickness
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors,
        x=node_x,
        y=node_y,
        hovertemplate='%{label}'
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values,
        label=link_labels,
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>',
        color='rgba(0,0,96,0.2)'  # Semi-transparent blue links
    ))])

# Step 6: Add Top 5 Learning Paths and Additional Statistics as Text Annotation

# Prepare the text with line breaks
formatted_text = full_text.replace('\n', '<br>')

# Adjust position to be below the diagram
fig.add_annotation(
    x=0.5,
    y=-0.10,  # Adjusted y position to place annotation below the diagram
    xref='paper',
    yref='paper',
    text=formatted_text,
    showarrow=False,
    font=dict(size=10),
    align='left',
    xanchor='center',
    yanchor='top',
    bordercolor='black',
    borderwidth=1,
    borderpad=10,
    bgcolor='white',
    opacity=0.8
)

# Update layout to adjust margins and set global font size
fig.update_layout(
    title_text="Learners' Journeys Sankey Diagram",
    font=dict(size=10),  # Set global font size to 10
    margin=dict(l=50, r=50, t=50, b=1500),  # Adjusted bottom margin to fit annotation
    height=2500,  # Adjusted height
    width=7500  # Adjusted width
)

# Step 7: Display or Save the Diagram

# To display the diagram in a browser window
fig.show()

# To save the diagram as an HTML file
fig.write_html("learning_paths_sankey.html")
