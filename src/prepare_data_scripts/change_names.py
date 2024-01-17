import os

img_path = '../../Original_untouched_dataset/resized/resized'

if __name__ == "__main__":
    for file_name in os.listdir(img_path):
        if file_name.startswith("Albrecht_DuÌrer"):
            # Extract the "_num" part of the file name
            num_part = file_name.split("_")[2]

            # Create the new file name with "Albrecht_Dürer" and the "_num" part
            new_file_name = f"Albrecht_Dürer_{num_part}"

            old_file_path = os.path.join(img_path, file_name)
            new_file_path = os.path.join(img_path, new_file_name)
            print(new_file_path)
            os.rename(old_file_path, new_file_path)
