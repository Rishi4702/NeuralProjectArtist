import os

img_path = "../../Original_untouched_dataset/resized/resized"

if __name__ == "__main__":
    for file_name in os.listdir(img_path):
        if file_name.startswith("Albrecht_Du¦êrer"):
            print(img_path + "/" + file_name)
            os.remove(img_path + "/" + file_name)
