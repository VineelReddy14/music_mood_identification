import pandas as pd

# File paths
csv_paths = {
    'complete': '/mnt/data/Vineel/jamendo_project/labels/moodtheme_labels.csv'
}

# Function to update .wav → .npy
def update_path_column(file_path):
    df = pd.read_csv(file_path)
    df['path'] = df['path'].str.replace('.wav', '.npy', regex=False)
    df.to_csv(file_path, index=False)

# Apply to each
for name, path in csv_paths.items():
    update_path_column(path)

print("Paths updated to .npy format.")
