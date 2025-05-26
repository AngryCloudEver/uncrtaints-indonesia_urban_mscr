import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load .npy dan ubah ke (H, W, C)
s1 = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-6_in.npy'), (1, 2, 0))
input = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-6_in.npy'), (1, 2, 0))
pred_dsen = np.asarray(Image.open('data/INDONESIA_URBAN_MSCR/For Comparison/6.png'))
pred_dsen_opt = np.asarray(Image.open('data/INDONESIA_URBAN_MSCR/For Comparison/6 Optimized.png'))
pred_uncrtaints_base = np.transpose(np.load('model/inference/monotemporalL2/export/epoch_1/test/img-6_pred.npy'), (1, 2, 0))
pred = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-6_pred.npy'), (1, 2, 0))
pred_retrain = np.transpose(np.load('model/results/monotemporalL2Indonesia/export/epoch_20/test/img-6_pred.npy'), (1, 2, 0))
target = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-6_target.npy'), (1, 2, 0))
pred_mask = Image.open('model/results/monotemporalL2/plots/epoch_40/test/img-6_mask.png')

width, height = pred_mask.size  # (640, 480)
new_width = new_height = 480

left = (width - new_width) // 2
top = (height - new_height) // 2
right = left + new_width
bottom = top + new_height

pred_mask = pred_mask.crop((left, top, right, bottom))
pred_mask = pred_mask.resize((256, 256), Image.NEAREST)
pred_mask = np.asarray(pred_mask)

# Ambil RGB (B4, B3, B2) dan ubah ke float32
s1_in = input[:, :, [1]].astype(np.float32)
rgb_in = input[:, :, [5, 4, 3]].astype(np.float32)
rgb_pred_base = pred_uncrtaints_base[:, :, [3, 2, 1]].astype(np.float32)
rgb_pred = pred[:, :, [3, 2, 1]].astype(np.float32)
rgb_pred_retrain = pred_retrain[:, :, [3, 2, 1]].astype(np.float32)
rgb_target = target[:, :, [3, 2, 1]].astype(np.float32)

brightness_factor = 4
rgb_in = np.clip(rgb_in * brightness_factor, 0, 1)
rgb_pred_base = np.clip(rgb_pred_base * brightness_factor, 0, 1)
rgb_pred_retrain = np.clip(rgb_pred_retrain * brightness_factor, 0, 1)
rgb_pred = np.clip(rgb_pred * brightness_factor, 0, 1)
rgb_target = np.clip(rgb_target * brightness_factor, 0, 1)

# Tampilkan berdampingan
plt.figure(figsize=(10, 12))

plt.subplot(3, 3, 1)
plt.imshow(rgb_in)
plt.title("Input (Cloudy)")
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(s1_in, cmap='gray')
plt.title("Input (Sentinel-1)")
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(pred_dsen)
plt.title("Dsen2-CR")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(pred_dsen_opt)
plt.title("Dsen2-CR Optimized")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(rgb_pred_base)
plt.title("UnCRtainTS Base")
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(rgb_pred_retrain)
plt.title("UnCRtainTS Retrained")
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(rgb_pred)
plt.title("UnCRtainTS PreTrained")
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(rgb_target)
plt.title("Target (Cloud-Free)")
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(pred_mask)
plt.title("Cloud Masking")
plt.axis('off')

plt.suptitle('Perbandingan Hasil Model Cloud Removal 1', fontsize=16)
plt.savefig('data/INDONESIA_URBAN_MSCR/For Comparison/Results/Comparison 1 (6).png', dpi=300)
plt.show()
