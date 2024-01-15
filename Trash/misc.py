import pandas as pd

# Load data from the CSV file
df = pd.read_csv('../src/Dataset/artists.csv')

# Initialize an empty genre-based dictionary
genre_dict = {}

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    # Split the 'genre' column by ',' to get a list of genres
    genres = row['genre'].split(',')

    # Iterate through the genres and add the 'name' to the corresponding genre's list
    for genre in genres:
        genre = genre.strip()  # Remove leading/trailing spaces
        if genre not in genre_dict:
            genre_dict[genre] = []
        genre_dict[genre].append(row['name'])

# Print the genre-based dictionary
print(genre_dict)
