import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

cloudy_dir = "data/TestingFull/30_Mei_2025_Makassar/Sentinel2_Mosaic_Makassar.tif"
sentinel1_dir = "data/TestingFull/30_Mei_2025_Makassar/Sentinel1_Mosaic_Makassar.tif"

output_cloudy = "data/TestingFull/30_Mei_2025_Makassar/Test/Sentinel-2-Cloudy/Makassar"
output_sentinel1 = "data/TestingFull/30_Mei_2025_Makassar/Test/Sentinel-1/Makassar"

os.makedirs(output_cloudy, exist_ok=True)
os.makedirs(output_sentinel1, exist_ok=True)

def extract_patches(image, patch_size=256):
    """Splits an image into 256x256 patches."""
    patches = []
    height, width = image.shape[1], image.shape[2]

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if i + patch_size <= height and j + patch_size <= width:
                patch = image[:, i:i+patch_size, j:j+patch_size]
                patches.append((patch, i, j))

    return patches

def load_tif(filepath, bands=None):
    """Loads selected bands from a .tif file using Rasterio."""
    with rasterio.open(filepath) as src:
        if bands:
            band_indexes = [src.descriptions.index(band) + 1 for band in bands]
            image = src.read(band_indexes)
        else:
            image = src.read()

    return image

output_dirs = {
    "cloudy": output_cloudy,
    "sentinel1": output_sentinel1
}

S2_BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S1_BAND_ORDER = ["VV", "VH"]

cloudy_img = load_tif(cloudy_dir, S2_BAND_ORDER)
sentinel1_img = load_tif(sentinel1_dir, S1_BAND_ORDER)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(np.clip(np.moveaxis(cloudy_img[[3, 2, 1], :, :], 0, -1) / 3000, 0, 1))
ax[0].set_title("Cloudy")
ax[1].imshow(sentinel1_img[0], cmap='gray')
ax[1].set_title("Sentinel 1")

plt.show()

grouped_files = {}

grouped_files.setdefault('Sentinel2_Mosaic_Makassar.tif', {})["cloudy"] = 'Sentinel2_Mosaic_Makassar.tif'
grouped_files.setdefault('Sentinel1_Mosaic_Makassar.tif', {})["sentinel1"] = 'Sentinel1_Mosaic_Makassar.tif'

print(grouped_files)

for base_name, files in grouped_files.items():
    cloudy_path = files.get("cloudy")
    sentinel1_path = files.get("sentinel1")

    if cloudy_img is None or sentinel1_img is None:
        continue  # Skip if any image is missing

    # Extract patches from images
    cloudy_patches = extract_patches(cloudy_img)
    sentinel1_patches = extract_patches(sentinel1_img)

    # Iterate over patches
    for idx, ((cl_patch, i, j), (s1_patch, _, _)) in enumerate(zip(cloudy_patches, sentinel1_patches)):
        patch_name = f"{base_name.replace('.tif', '')}_{idx+1}.tif"

        # Save Cloudy Image (Tanpa Metadata)
        cl_save_path = os.path.join(output_dirs["cloudy"], patch_name)
        with rasterio.open(cl_save_path, "w", driver="GTiff", height=256, width=256, count=cl_patch.shape[0], dtype=cl_patch.dtype) as dst:
            dst.write(cl_patch)

        # Save Sentinel-1 Image (Tanpa Metadata)
        s1_save_path = os.path.join(output_dirs["sentinel1"], patch_name)
        with rasterio.open(s1_save_path, "w", driver="GTiff", height=256, width=256, count=s1_patch.shape[0], dtype=s1_patch.dtype) as dst:
            dst.write(s1_patch)

        print(f"Saved {patch_name} (Cloudy, Sentinel-1)")