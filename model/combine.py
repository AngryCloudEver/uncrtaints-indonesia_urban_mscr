import os
import re
from PIL import Image, ImageEnhance

def reconstruct_image_from_patches(folder_path, width, height, save_dir, save_name, patch_size=256):
    # Ambil semua file img-[x]_pred.png
    pattern = re.compile(r"img-(\d+)_pred\.png")
    files = [
        f for f in os.listdir(folder_path)
        if pattern.match(f)
    ]
    
    # Urutkan berdasarkan angka x
    files_sorted = sorted(files, key=lambda x: int(pattern.match(x).group(1)))

    # Hitung jumlah patch horizontal dan vertikal
    cols = width // patch_size
    rows = height // patch_size

    print(f"Reconstructing image with {rows} rows and {cols} cols")

    # Buat array kosong
    reconstructed = Image.new("RGB", (cols * patch_size, rows * patch_size))

    for idx, filename in enumerate(files_sorted):
        patch = Image.open(os.path.join(folder_path, filename))

        row = idx // cols
        col = idx % cols

        reconstructed.paste(patch, (col * patch_size, row * patch_size))

    enhancer = ImageEnhance.Brightness(reconstructed)
    brightened = enhancer.enhance(3.5)

    # Simpan ke direktori tujuan
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        brightened.save(save_path)
        print(f"Reconstructed image saved to: {save_path}")

    return brightened

folder = r"model\TestingFull\28_April_2025_Makassar\Inference\monotemporalL2\plots\epoch_1\test"
save_folder = r"data\TestingFull\28_April_2025_Makassar\Result"

width = 2034
height = 2887

output_img = reconstruct_image_from_patches(folder, width, height, save_folder, 'Makassar Result.png')
output_img.show()