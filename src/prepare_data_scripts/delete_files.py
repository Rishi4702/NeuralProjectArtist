import os

img_path = "../../dataset_files/resized"

if __name__ == "__main__":
    for file_name in os.listdir(img_path):
        if file_name.startswith("Albrecht_DuÌrer"):
            os.remove(img_path + "/" + file_name)