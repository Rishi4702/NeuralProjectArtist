from Dataloader.ArtDataset import ArtDataset
import os

dataset = ArtDataset(csv_file='../Dataset/artists.csv', img_dir='../Dataset/resized')
# Assuming 'dataset' is your ArtDataset instance
print("Dataset size:", len(dataset))

# Get the first data sample
image, artist_data = dataset[0]

# Check image and artist data
print("Image type:", type(image))
print("Artist Data:", artist_data)

for i in range(len(dataset)):
    image, artist_data = dataset[i]
    img_file_name = os.listdir(dataset.img_dir)[i]
    artist_name = artist_data['name']
    counter += 1
    #print(f"Image File: {img_file_name} - Painter: {artist_data}")
