import pandas as pd

# Load the CSV file
df = pd.read_csv(r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\Original_untouched_dataset\resized\csv\artists.csv')

# Keep only 'name', 'genre', and 'paintings' columns
df = df[['name', 'genre', 'paintings']]

# Keep only the first genre for each artist
df['genre'] = df['genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)

# Group by 'genre' and sum the 'paintings'
genre_counts = df.groupby('genre')['paintings'].sum().sort_values(ascending=False)

# Get the top 5 genres
top_5_genres = genre_counts.head(10).index.tolist()
print(top_5_genres)

# Filter the DataFrame to keep only artists that belong to one of the top 5 genres
df_top_5_genres = df[df['genre'].isin(top_5_genres)]
print(df_top_5_genres)
# Save the processed DataFrame to a new CSV file
df_top_5_genres.to_csv(r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\csv\top_5_artists_genres.csv', index=False)
