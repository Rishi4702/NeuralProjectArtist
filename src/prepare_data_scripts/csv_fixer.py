import os.path
import shutil

import pandas as pd

# Load the CSV file
df = pd.read_csv("../../Original_untouched_dataset/resized/csv/artists.csv")
df = df[["name", "genre", "paintings"]]
df["genre"] = df["genre"].apply(lambda x: x.split(",")[0] if pd.notnull(x) else x)

genre_counts = df.groupby("genre")["paintings"].sum().sort_values(ascending=False)
top_5_genres = genre_counts.head(10).index.tolist()
df_top_5_genres = df[df["genre"].isin(top_5_genres)]
df_top_5_genres.to_csv("../../dataset_files/csv/artists.csv", index=False)

original_data_path = "../../Original_untouched_dataset/resized/resized"
files = os.listdir(original_data_path)

df_top_5_genres = [name.replace(" ", "_") for name in list(df_top_5_genres["name"])]
count = 0

for file in files:
    names = file.split("_")
    name = "_".join(names[:-1])

    if name in df_top_5_genres:
        count += 1
        src = os.path.join(original_data_path, file)
        dst = os.path.join("../../dataset_files/resized", file)

        shutil.copyfile(src, dst)

print(count)