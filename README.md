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

## Prepare dataset and environment

This code is tested on Ubuntu 18.04 with Python 3.7, CUDA 11.0 and Pytorch 1.7

1, Install the following dependencies by `pip install -r requirements.txt`.

2, Download TAU-NIGENS Spatial Sound Events 2021 dataset [here](https://zenodo.org/record/4844825). 
This code also works with TAU-NIGENS Spatial Sound Events 2020 dataset [here](https://zenodo.org/record/4064792). 

3, Extract everything into the same folder. 

4, Data file structure should look like this:

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

todo

## Training

todo

## Evaluate our pretrained model

todo

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
