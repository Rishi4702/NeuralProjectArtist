import os

img_path = "../../Dataset/resized"

for file_name in os.listdir(img_path):
    if file_name.startswith("Albrecht_DuÌrer"):
        os.remove(img_path + "/" + file_name)
