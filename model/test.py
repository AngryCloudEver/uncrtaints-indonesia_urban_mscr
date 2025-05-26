import numpy as np
import matplotlib.pyplot as plt

# Load .npy dan ubah ke (H, W, C)
s1 = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-37_in.npy'), (1, 2, 0))
input = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-37_in.npy'), (1, 2, 0))
pred = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-37_pred.npy'), (1, 2, 0))
target = np.transpose(np.load('model/results/monotemporalL2/export/epoch_40/test/img-37_target.npy'), (1, 2, 0))

# Ambil RGB (B4, B3, B2) dan ubah ke float32
s1_in = input[:, :, [1]].astype(np.float32)
rgb_in = input[:, :, [5, 4, 3]].astype(np.float32)
rgb_pred = pred[:, :, [3, 2, 1]].astype(np.float32)
rgb_target = target[:, :, [3, 2, 1]].astype(np.float32)

brightness_factor = 2.5
rgb_in = np.clip(rgb_in * brightness_factor, 0, 1)
rgb_pred = np.clip(rgb_pred * brightness_factor, 0, 1)
rgb_target = np.clip(rgb_target * brightness_factor, 0, 1)

# Tampilkan berdampingan
plt.figure(figsize=(6, 6))

plt.subplot(2, 2, 1)
plt.imshow(rgb_in)
plt.title("Input (Cloudy)")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(s1_in, cmap='gray')
plt.title("Input (Sentinel-1)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(rgb_pred)
plt.title("Prediction (UnCRtainTS)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(rgb_target)
plt.title("Target (Cloud-Free)")
plt.axis('off')

plt.tight_layout()
plt.show()
