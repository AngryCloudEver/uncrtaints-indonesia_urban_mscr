import os
import glob
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted, humansorted
import subprocess as sub

from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector

import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return img

def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask


# recursively apply function to nested dictionary
def iterdict(dictionary, fct):
    for k,v in dictionary.items():        
        if isinstance(v, dict):
            dictionary[k] = iterdict(v, fct)
        else:      
            dictionary[k] = fct(v)      
    return dictionary

def get_cloud_map(img, detector, instance=None):

    # get cloud masks
    img = np.clip(img, 0, 10000)
    mask = np.ones((img.shape[-1], img.shape[-1]))
    # note: if your model may suffer from dark pixel artifacts,
    #       you may consider adjusting these filtering parameters
    if not (img.mean()<1e-5 and img.std() <1e-5):
        if detector == 'cloud_cloudshadow_mask':
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = get_cloud_cloudshadow_mask(img, threshold)
        elif detector== 's2cloudless_map':
            threshold = 0.5
            mask = instance.get_cloud_probability_maps(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2)
        elif detector == 's2cloudless_mask':
            mask = instance.get_cloud_masks(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
        else:
            mask = np.ones((img.shape[-1], img.shape[-1]))
            warnings.warn(f'Method {detector} not yet implemented!')
    else:   warnings.warn(f'Encountered a blank sample, defaulting to cloudy mask.')
    return mask.astype(np.float32)


# function to fetch paired data, which may differ in modalities or dates
def get_pairedS1(patch_list, root_dir, mod=None, time=None):
    paired_list = []
    for patch in patch_list:
        seed, roi, modality, time_number, fname = patch.split('/')
        time = time_number if time is None else time # unless overwriting, ...
        mod  = modality if mod is None else mod      # keep the patch list's original time and modality
        n_patch       = fname.split('patch_')[-1].split('.tif')[0]
        paired_dir    = os.path.join(seed, roi, mod.upper(), str(time))
        candidates    = os.path.join(root_dir, paired_dir, f'{mod}_{seed}_{roi}_ImgNo_{time}_*_patch_{n_patch}.tif')
        paired_list.append(os.path.join(paired_dir, os.path.basename(glob.glob(candidates)[0])))
    return paired_list

def get_paired_data(patch_list, root_dir, mod=None):
    paired_list = []
    
    if mod not in ['s1', 's2']:
        raise ValueError("Modality harus 's1' untuk Sentinel-1 atau 's2' untuk Sentinel-2 Cloudy")

    for patch in patch_list:
        seed, roi, fname = patch.split('/')[-3:]  # Ambil hanya bagian akhir dari path
        
        # Tentukan direktori berdasarkan modality
        if mod == 's1':
            paired_dir = os.path.join(root_dir, 'Test/Sentinel-1', roi)
        elif mod == 's2':
            paired_dir = os.path.join(root_dir, 'Test/Sentinel-2-Cloudy', roi)
        
        paired_path = os.path.join(paired_dir, fname)
        paired_list.append(paired_path)
    
    return paired_list


""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    depricated --> vary_samples:       bool, whether to draw random samples across epochs or not, matters only if sample_type is 'cloud_cloudfree'
    sampler             str, [fixed | fixedsubset | random]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    min_cov:            float, in [0.0, 1.0]
    max_cov:            float, in [0.0, 1.0]
    import_data_path:   str, path to importing the suppl. file specifying what time points to load for input and output
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

""" SEN12MSCR data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class SEN12MSCR(Dataset):
    def __init__(self, root, split="all", cloud_masks='s2cloudless_mask', sample_type='pretrain', rescale_method='default'):

        self.root_dir = root                                # set root directory which contains all ROI
        
        # define splits conform with SEN12MS-CR-TS
        self.splits          = {}
        self.splits['train'] = ['Train/Sentinel-1/Jakarta']
        self.splits['val']   = ['Val/Sentinel-1/Jakarta'] 
        self.splits['test']  = ['Test/Sentinel-1/Jakarta']

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['pretrain'], "Input data must be pretrain!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.cloud_masks    = cloud_masks   # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type   # e.g. 'pretrain'

        self.time_points    = range(1)
        self.n_input_t      = 1             # specifies the number of samples, if only part of the time series is used as an input

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else: self.cloud_detector = None

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

        self.method         = rescale_method

    def getsample(self, pdx):
        return self.__getitem__(pdx)

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region all')

        paths = []

        if self.split == 'train':
            path_S1 = humansorted([s1dir for s1dir in os.listdir(self.root_dir + '/Train/Sentinel-1')])
        elif self.split == 'val':
            path_S1 = humansorted([s1dir for s1dir in os.listdir(self.root_dir + '/Val/Sentinel-1')])
        elif self.split == 'test':
            path_S1 = humansorted([s1dir for s1dir in os.listdir(self.root_dir + '/Test/Sentinel-1')])
        else:
            path_S1 = []
            for subset in ['Train', 'Val', 'Test']:
                subset_path = os.path.join(self.root_dir, subset, 'Sentinel-1')
                if os.path.exists(subset_path):
                    path_S1.extend(humansorted(os.listdir(subset_path)))

        for location in tqdm(path_S1):
            base_dir = os.path.join(self.root_dir, self.split.capitalize(), 'Sentinel-1', location)
            paths_S1 = natsorted([os.path.join(base_dir, s1patch) for s1patch in os.listdir(base_dir)])
            paths_S2 = [patch.replace('Sentinel-1', 'Sentinel-2-CloudFree') for patch in paths_S1]
            paths_S2_cloudy = [patch.replace('Sentinel-1', 'Sentinel-2-Cloudy') for patch in paths_S1]


            for pdx, _ in enumerate(paths_S1):
                # omit patches that are potentially unpaired
                if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx]), os.path.isfile(paths_S2_cloudy[pdx])]): continue
                # don't add patch if not belonging to the selected split
                # if not any([split_roi in paths_S1[pdx] for split_roi in self.splits[self.split]]): continue
                sample = {"S1":         paths_S1[pdx],
                            "S2":         paths_S2[pdx],
                            "S2_cloudy":  paths_S2_cloudy[pdx]}
                paths.append(sample)
                
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(self.paths[pdx]['S1'])
        s2_tif          = read_tif(self.paths[pdx]['S2'])
        s2_cloudy_tif   = read_tif(self.paths[pdx]['S2_cloudy'])
        coord           = list(s2_tif.bounds)
        s1              = process_SAR(read_img(s1_tif), self.method)
        s2              = read_img(s2_tif)           # note: pre-processing happens after cloud detection
        s2_cloudy       = read_img(s2_cloudy_tif)    # note: pre-processing happens after cloud detection
        mask            = None if not self.cloud_masks else get_cloud_map(s2_cloudy, self.cloud_masks, self.cloud_detector)

        sample = {'input': {'S1': s1,
                            'S2': process_MS(s2_cloudy, self.method),
                            'masks': mask,
                            'coverage': np.mean(mask),
                            'S1 path': os.path.join(self.root_dir, self.paths[pdx]['S1']),
                            'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']),
                            'coord': coord,
                            },
                    'target': {'S2': process_MS(s2, self.method),
                               'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2']),
                               'coord': coord,
                                },
                    }
        return sample

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:

        path/to/your/SEN12MSCR/directory:
            ├───ROIs1158_spring_s1
            |   ├─s1_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s1_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2
            |   ├─s2_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2_cloudy
            |   ├─s2_cloudy_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_cloudy_1_p407.tif
            |   |   |...
            |    ...
            ...

        Note: Please arrange the dataset in a format as e.g. provided by the script dl_data.sh.
        """)

    def __len__(self):
        # length of generated list
        return self.n_samples