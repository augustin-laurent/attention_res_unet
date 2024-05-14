import os
import shutil
from PIL import Image

def is_totally_black(image_path, threshold=10):
    """
    Check if an image is totally black.
    """
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        pixels = list(img_gray.getdata())
        return all(p < threshold for p in pixels)
    
def move_black_images(folder_path, dest_folder, threshold=10):
    """
    Move black images from a folder to another folder.
    """
    os.makedirs(dest_folder, exist_ok=True)
    for file in os.listdir(folder_path):
        if is_totally_black(os.path.join(folder_path, file), threshold):
            shutil.move(os.path.join(folder_path, file), os.path.join(dest_folder, file))

def sync_files(folder_a: str, folder_b: str, folder_c: str):
    files_in_c = set(os.listdir(folder_c))
    os.makedirs(folder_b, exist_ok=True)
    for file in os.listdir(folder_a):
        if file in files_in_c:
            src_file = os.path.join(folder_a, file)
            dst_file = os.path.join(folder_b, file)
            shutil.move(src_file, dst_file)

#move_black_images(folder_path="/mnt/c/Users/augus/Downloads/patches_erg_512", dest_folder="/mnt/c/Users/augus/Downloads/ERG_no_cell", threshold=10)
print("Start syncing images for HE Folder")
sync_files("/mnt/c/Users/augus/Downloads/patches_he_512", "/mnt/c/Users/augus/Downloads/HE_no_cell", "/mnt/c/Users/augus/Downloads/ERG_no_cell")