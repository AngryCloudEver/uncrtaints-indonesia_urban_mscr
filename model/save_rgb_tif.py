import os
import tifffile
import numpy as np
from PIL import Image, ImageEnhance

def convert_single_tif_to_rgb_png(tif_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Baca file .tif
    img = tifffile.imread(tif_path)

    # Ubah ke HWC kalau masih CHW
    if img.ndim == 3 and img.shape[0] < 20:
        img = np.transpose(img, (1, 2, 0))

    # Ambil channel 3,2,1 (index 3,2,1) untuk RGB
    try:
        rgb = img[:, :, [3, 2, 1]]
    except IndexError:
        print(f"[ERROR] File {tif_path} tidak punya cukup band.")
        return

    # Normalisasi ke uint8 jika perlu
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)

    # Konversi ke PIL Image dan tambah brightness
    img_rgb = Image.fromarray(rgb)
    enhancer = ImageEnhance.Brightness(img_rgb)
    bright_img = enhancer.enhance(3.5)

    # Simpan hasil
    save_path = os.path.join(save_dir, 'Makassar Cloudy.png')
    bright_img.save(save_path)
    print(f"✅ Saved: {save_path}")

# Contoh pemakaian:
tif_path = r"data\TestingFull\28_April_2025_Makassar\Sentinel2_Mosaic_Makassar.tif"
save_dir = r"data\TestingFull\28_April_2025_Makassar\Result"

convert_single_tif_to_rgb_png(tif_path, save_dir)
