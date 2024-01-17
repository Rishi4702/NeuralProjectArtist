import pandas as pd

# Load the CSV file
df = pd.read_csv(r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\Original_untouched_dataset\resized\csv\artists.csv')

# Keep only 'name' and 'genre' columns
df = df[['name', 'genre']]

# Keep only the first genre for each row
df['genre'] = df['genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)


# Save the processed DataFrame to a new CSV file
df.to_csv('../../dataset_files/csv/artist.csv', index=False)

