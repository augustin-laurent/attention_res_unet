import os
import shutil

def build_eval_set(src_dir: str, dst_dir: str):
    image_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    image_files.sort(key=lambda f: f.split('_')[0])
    highest_id = image_files[-1].split('_')[0]

    os.makedirs(dst_dir, exist_ok=True)

    for f in image_files:
        if f.startswith(highest_id):
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

def sync_files(folder_a: str, folder_b: str, folder_c: str):
    files_in_c = set(os.listdir(folder_c))

    os.makedirs(folder_b, exist_ok=True)
    for file in os.listdir(folder_b):
        if file in files_in_c:
            src_file = os.path.join(folder_a, file)
            dst_file = os.path.join(folder_b, file)
            shutil.move(src_file, dst_file)

print("Start spliting mask")
build_eval_set("/mnt/c/Users/augus/Downloads/patches_erg_512", "/mnt/c/Users/augus/Downloads/patches_erg_512_eval")
print("Splitting images")
sync_files("/mnt/c/Users/augus/Downloads/patches_he_512", "/mnt/c/Users/augus/Downloads/patches_he_512_eval", "/mnt/c/Users/augus/Downloads/patches_erg_512_eval")
print("Splitting masks")