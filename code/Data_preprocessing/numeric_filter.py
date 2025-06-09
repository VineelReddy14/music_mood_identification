import pandas as pd
import numpy as np
import ast

# Define file paths
input_files = {
    'complete': '/mnt/data/Vineel/jamendo_project/labels/autotagging_moodtheme_multihot.csv'
}

output_files = {
    'complete': '/mnt/data/Vineel/jamendo_project/labels/moodtheme_labels.csv'
}

# Function to process and save the file
def process_and_save(input_file, output_file):
    df = pd.read_csv(input_file)

    # Convert string representation of lists to actual numeric lists
    tags_expanded = df['tags'].apply(lambda x: np.array(ast.literal_eval(x)))

    # Convert lists to a DataFrame where each element becomes a separate column
    tags_df = pd.DataFrame(tags_expanded.tolist())

    # Combine track_id, path, and expanded tags
    final_df = pd.concat([df[['track_id', 'path']], tags_df], axis=1)

    # Save to CSV
    final_df.to_csv(output_file, index=False)

# Process each file
for key in input_files:
    process_and_save(input_files[key], output_files[key])

print("Conversion completed successfully.")
