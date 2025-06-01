import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

S1_BAND_ORDER = ["VV", "VH"]

def load_tif(filepath, bands=None):
    """Loads selected bands from a .tif file using Rasterio."""
    with rasterio.open(filepath) as src:
        if bands:
            band_indexes = [src.descriptions.index(band) + 1 for band in bands]
            image = src.read(band_indexes)
        else:
            image = src.read()
    return image

sentinel1_dir = "data/TestingFull/30_Mei_2025_Makassar/Sentinel1_Mosaic_Makassar.tif"

# Load prediction result (target) first to get size
target = np.asarray(Image.open('data/TestingFull/30_Mei_2025_Makassar/Result/Makassar Result.png'))
target_height, target_width = target.shape[:2]

# Load cloudy input and resize
input_img = Image.open('data/TestingFull/30_Mei_2025_Makassar/Result/Makassar Cloudy.png')
input_resized = input_img.resize((target_width, target_height), Image.BICUBIC)
input_array = np.asarray(input_resized)

# Load Sentinel-1 (VV) and resize
s1 = load_tif(sentinel1_dir, S1_BAND_ORDER)
s1_resized = []
for band in s1:
    band_img = Image.fromarray(band)
    resized_band = band_img.resize((target_width, target_height), Image.BICUBIC)
    s1_resized.append(np.asarray(resized_band))
s1_resized = np.stack(s1_resized)

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.imshow(s1_resized[0], cmap='gray')
plt.title("Input (Sentinel 1 VV)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(input_array)
plt.title("Input (Cloudy)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(target)
plt.title("Prediction (UnCRtainTS)")
plt.axis('off')

plt.tight_layout()
plt.suptitle('Hasil Model Cloud Removal Makassar - 30 Mei 2025', fontsize=16)
plt.savefig('data/TestingFull/30_Mei_2025_Makassar/Result/Makassar Comparison.png', dpi=300)
plt.show()
