# SALSA: Spatial Cue-AugmentedLog-Spectrogram Features for Polyphonic Sound Event Localization and Detection

<p align="center">
        <img src="figures/foa_salsa_16s_tight_with_text.png" title="SALSA feature for first-order ambisonics microphone (FOA)" width="48%"> 
        <img src="figures/mic_salsa_16s_tight_with_text.png" title="SALSA feature for microphone array (MIC)" width="48%">
        <em>Visualization of SALSA features of a 16-second audio clip in multi-source scenarios for 
            first-order ambisonics microphone (FOA) (left) and 4-channel microphone array (MIC) (right).</em>
</p>

Official implementation for Spatial Cue-AugmentedLog-Spectrogram Features **SALSA** for polyphonic sound event localization and detection.

<pre>
SALSA: Spatial Cue-AugmentedLog-Spectrogram Features for Polyphonic Sound Event Localization and Detection
Thi Ngoc Tho Nguyen; Karn N. Watcharasupat; Ngoc Khanh Nguyen; Douglas L. Jones; Woon-Seng Gan. 
</pre>

[[**ArXiv paper**]](https://arxiv.org/xxx)

## What is SALSA?

todo

## Network architecture

todo

## Prepare dataset and environment

This code is tested on Ubuntu 18.04 with Python 3.7, CUDA 11.0 and Pytorch 1.7

1. Install the following dependencies by `pip install -r requirements.txt`. Or manually install these modules:
    * numpy
    * scipy
    * pandas
    * scikit-learn
    * h5py
    * librosa
    * tqdm
    * pytorch 1.7
    * pytorch-lightning      
    * tensorboardx
    * pyyaml
    * einops

2. Download TAU-NIGENS Spatial Sound Events 2021 dataset [here](https://zenodo.org/record/4844825). 
This code also works with TAU-NIGENS Spatial Sound Events 2020 dataset [here](https://zenodo.org/record/4064792). 

3. Extract everything into the same folder. 

4. Data file structure should look like this:

```
./
├── feature_extraction.py
├── ...
└── data/
    ├──foa_dev
    │   ├── fold1_room1_mix001.wav
    │   ├── fold1_room1_mix002.wav  
    │   └── ...
    ├──foa_eval
    ├──metadata_dev
    ├──metadata_eval (might not be available yet)
    ├──mic_dev
    └──mic_eval
```

## Feature extraction

Our code support the following features:  

| Name        | Format   | Component     | Number of channels |
| :---        | :----:   | :---          |  :----:            |
| melspeciv   | FOA      | multichannel log-mel spectrograms  + intensity vector    | 7 |
| linspeciv   | FOA      | multichannel log-linear spectrograms  + intensity vector    | 7 |
| melspecgcc  | MIC      | multichannel log-mel spectrograms  + GCC-PHAT    | 10 |
| linspecgcc  | MIC      | multichannel log-linear spectrograms  + GCC-PHAT   | 10 |
| **SALSA**   | FOA      | multichannel log-linear spectrograms  + eigenvector-based intensity vector (EIV)    | 7 |
| **SALSA**   | MIC      | multichannel log-linear spectrograms  + eigenvector-based phase vector (EPV)    | 7 |

Note: the number of channels are calculated based on four-channel inputs.

To extract **SALSA** feature, edit directories for data and feature accordingly in `tnsse_2021_salsa_feature_config.yml` in 
`dataset\configs\` folder. Then run `make salsa`

To extract *linspeciv*, *melspeciv*, *linspecgcc*, *melspecgcc* feature, 
edit directories for data and feature accordingly in `tnsse_2021_feature_config.yml` in 
`dataset\configs\` folder. Then run `make feature`

## Training and inference

To train SELD model with SALSA feature, edit the *feature_root_dir* and *gt_meta_root_dir* in the experiment config 
`experiments\configs\seld.yml`. Then run `make train`. 

To do inference, run `make inference`. To evaluate output, edit the `Makefile` accordingly and run `make evaluate`.

## Citation
Please consider citing our paper if you find this code useful for your research. Thank you!!!
```
@InProceedings{tho2021salsa,
author = {Nguyen, Thi Ngoc Tho and Watcharasupat, Karn N. and Nguyen, Ngoc Khanh and Jones, Douglas L. and Gan, Woon-Seng},
title = {SALSA: Spatial Cue-AugmentedLog-Spectrogram Features for Polyphonic Sound Event Localization and Detection},
booktitle = {xxx},
month = {xxx},
year = {2021}
}
```
