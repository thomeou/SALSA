"""
This module extract salsa features for both first order ambisonics and microphone array formats.
"""
import os
import shutil

import fire
import h5py
import librosa
import numpy as np
import yaml
from sklearn import preprocessing
from timeit import default_timer as timer
from tqdm import tqdm


def extract_normalized_eigenvector(X, condition_number: float = 5.0, n_hopframes: int = 3, is_tracking: bool = True,
                                   audio_format: str = 'foa', fs: int = None, n_fft: int = None, lower_bin: int = None):
    """
    Function to extract normalized eigenvector.
    :param X: <np.ndarray (n_bins, n_frames, n_chans): Clipped spectrogram between lower_bin and upper_bin.
    :param is_tracking: If True, use noise-floor tracking
    :param audio_format: Choice: 'foa' (take real part)| 'mic' (take phase)
    """
    # get size of X
    n_bins, n_frames, n_chans = X.shape

    # noise floor tracking params
    n_sig_frames = 3
    indicator_countdown = n_sig_frames * np.ones((n_bins,), dtype=int)
    alpha = 0.02
    slow_scale = 0.1
    floor_up = 1 + alpha
    floor_up_slow = 1 + slow_scale * alpha
    floor_down = 1 - alpha
    snr_ratio = 1.5

    if audio_format == 'mic':
        c = 343
        delta = 2 * np.pi * fs / (n_fft * c)

    # padding X for Rxx computation
    X = np.pad(X, ((0, 0), (n_hopframes, n_hopframes), (0, 0)), 'wrap')

    # select type of signal for bg noise tracking:
    ismag = 0  # 0 use running average for tracking, 1 use raw magnitude for tracking
    signal_magspec = np.zeros((n_bins, n_frames))
    # signal to track
    n_autocorr_frames = 3
    if ismag == 1:
        signal_magspec = np.abs(X[:, n_hopframes:n_hopframes + n_frames, 0])
    else:
        for iframe in np.arange(n_autocorr_frames):
            signal_magspec = signal_magspec + np.abs(X[:, n_hopframes - iframe:n_hopframes - iframe + n_frames, 0]) ** 2
        signal_magspec = np.sqrt(signal_magspec / n_autocorr_frames)

    # Initial noisefloor assuming first few frames are noise
    noise_floor = 0.5*np.mean(signal_magspec[:, 0:5], axis=1)

    # memory to store output
    normalized_eigenvector_mat = np.zeros((n_chans - 1, n_bins, n_frames))  # normalized eigenvector of ss tf bin
    # =========================================================================
    for iframe in np.arange(n_hopframes, n_frames + n_hopframes):
        # get current frame tracking singal
        xfmag = signal_magspec[:, iframe - n_hopframes]
        # ---------------------------------------------------------------------
        # bg noise tracking: implement direct up/down noise floor tracker
        above_noise_idx = xfmag > noise_floor
        # ------------------------------------
        # if signal above noise floor
        indicator_countdown[above_noise_idx] = indicator_countdown[above_noise_idx] - 1
        negative_indicator_idx = indicator_countdown < 0
        # update noise slow for bin above noise and negative indicator
        an_ni_idx = np.logical_and(above_noise_idx, negative_indicator_idx)
        noise_floor[an_ni_idx] = floor_up_slow * noise_floor[an_ni_idx]
        # update noise for bin above noise and positive indicator
        an_pi_idx = np.logical_and(above_noise_idx, np.logical_not(negative_indicator_idx))
        noise_floor[an_pi_idx] = floor_up * noise_floor[an_pi_idx]
        # reset indicator counter for bin below noise floor
        indicator_countdown[np.logical_not(above_noise_idx)] = n_sig_frames
        # reduce noise floor for bin below noise floor
        noise_floor[np.logical_not(above_noise_idx)] = floor_down * noise_floor[np.logical_not(above_noise_idx)]
        # make sure noise floor does not go to 0
        noise_floor[noise_floor < 1e-6] = 1e-6
        # --------------------------------------
        # select TF bins above noise level
        indicator_sig = xfmag > (snr_ratio * noise_floor)
        # ---------------------------------------------------------------------
        # valid bin after onset and noise background tracking
        if is_tracking:
            valid_bin = indicator_sig
        else:
            valid_bin = np.ones((n_bins,), dtype='bool')
        # ---------------------------------------------------------------------
        # coherence test
        for ibin in np.arange(n_bins):
            if valid_bin[ibin]:
                # compute covariance matrix using (2*nframehop + 1) frames
                X1 = X[ibin, iframe - n_hopframes:iframe + n_hopframes + 1, :]  # (2*n_hopframes+1) x nchan
                Rxx1 = np.dot(X1.T, X1.conj()) / float(2 * n_hopframes + 1)

                # svd: u: n_chans x n_chans, s: n_chans, columns of u is the singular vectors
                u, s, v = np.linalg.svd(Rxx1)

                # coherence test
                if s[0] > s[1] * condition_number:
                    indicator_rank1 = True
                else:
                    indicator_rank1 = False
                # update valid bin
                if is_tracking:
                    valid_bin[ibin] = valid_bin[ibin] and indicator_rank1

                # compute doa spectrum
                if valid_bin[ibin]:
                    # normalize largest eigenvector
                    if audio_format == 'foa':
                        normed_eigenvector = np.real(u[1:, 0] / u[0, 0])
                        normed_eigenvector = normed_eigenvector/np.sqrt(np.sum(normed_eigenvector**2))
                    elif audio_format == 'mic':
                        normed_eigenvector = np.angle(u[1:, 0] * np.conj(u[0, 0]))  # get the phase difference
                        # normalized for the frequency and delta
                        normed_eigenvector = normed_eigenvector / (delta * (ibin + lower_bin))
                    else:
                        raise ValueError('audio format {} is not valid'.format(audio_format))
                    # save output
                    normalized_eigenvector_mat[:, ibin, iframe - n_hopframes] = normed_eigenvector

    return normalized_eigenvector_mat


class MagStftExtractor:
    """
    Extract single-channel or multi-channel log-linear spectrograms. return feature of shape 4 x n_timesteps x 200
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int = None, window: str = 'hann',
                 is_compress_high_freq: bool = True):
        """
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param window: Type of window.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        if win_length is None:
            self.win_length = self.n_fft
        else:
            self.win_length = win_length
        assert self.win_length <= self.n_fft, 'Windown length is greater than nfft!'
        assert n_fft == 512 or n_fft == 256, 'nfft is not 512 or 256'
        if is_compress_high_freq:
            if n_fft == 512:
                self.W = np.zeros((200, 257), dtype=np.float32)
                for i in np.arange(192):
                    self.W[i, i+1] = 1.0
                for i in np.arange(192, 200):
                    if i < 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 8] = 1/8
                    elif i == 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 7] = 1/8
            elif n_fft == 256:
                self.W = np.zeros((100, 129), dtype=np.float32)
                for i in np.arange(96):
                    self.W[i, i+1] = 1.0
                for i in np.arange(96, 100):
                    if i < 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 8] = 1/8
                    elif i == 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 7] = 1/8
        else:
            self.W = np.zeros((n_fft // 2, n_fft // 2 + 1), dtype=np.float32)
            for i in np.arange(n_fft // 2):
                self.W[i, i + 1] = 1.0

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        :return: logmel_features <np.ndarray: (n_channels, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        log_features = []

        for i_channel in range(n_channels):
            spec = np.abs(librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                                       n_fft=self.n_fft,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length,
                                       center=True,
                                       window=self.window,
                                       pad_mode='reflect'))

            spec = np.dot(self.W, spec**2).T
            log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
            log_spec = np.expand_dims(log_spec, axis=0)
            log_features.append(log_spec)

        log_features = np.concatenate(log_features, axis=0)

        return log_features


def compute_scaler(feature_dir: str, audio_format: str) -> None:
    """
    Compute feature mean and std vectors of spectrograms for normalization.
    :param feature_dir: Feature directory that contains train and test folder.
    :param audio_format: Audio format, either 'foa' or 'mic'
    """
    print('============> Start calculating scaler')
    start_time = timer()

    # Get list of feature filenames
    train_feature_dir = os.path.join(feature_dir, audio_format + '_dev')
    feature_fn_list = os.listdir(train_feature_dir)

    # Get the dimensions of feature by reading one feature files
    full_feature_fn = os.path.join(train_feature_dir, feature_fn_list[0])
    with h5py.File(full_feature_fn, 'r') as hf:
        afeature = hf['feature'][:]  # (n_chanels, n_timesteps, n_features)
    n_channels = afeature.shape[0]
    assert n_channels == 7, 'only support n_channels = 7, got {}'.format(n_channels)
    n_feature_channels = 4  # hard coded number

    # initialize scaler
    scaler_dict = {}
    for i_chan in np.arange(n_feature_channels):
        scaler_dict[i_chan] = preprocessing.StandardScaler()

    # Iterate through data
    for count, feature_fn in enumerate(tqdm(feature_fn_list)):
        full_feature_fn = os.path.join(train_feature_dir, feature_fn)
        with h5py.File(full_feature_fn, 'r') as hf:
            afeature = hf['feature'][:]  # (n_chanels, n_timesteps, n_features)
            for i_chan in range(n_feature_channels):
                scaler_dict[i_chan].partial_fit(afeature[i_chan, :, :])  # (n_timesteps, n_features)


    # Extract mean and std
    feature_mean = []
    feature_std = []
    for i_chan in range(n_feature_channels):
        feature_mean.append(scaler_dict[i_chan].mean_)
        feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)

    # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
    feature_mean = np.expand_dims(feature_mean, axis=1)
    feature_std = np.expand_dims(feature_std, axis=1)

    scaler_path = os.path.join(feature_dir, audio_format + '_feature_scaler.h5')
    with h5py.File(scaler_path, 'w') as hf:
        hf.create_dataset('mean', data=feature_mean, dtype=np.float32)
        hf.create_dataset('std', data=feature_std, dtype=np.float32)

    print('Features shape: {}'.format(afeature.shape))
    print('mean {}: {}'.format(feature_mean.shape, feature_mean))
    print('std {}: {}'.format(feature_std.shape, feature_std))
    print('Scaler path: {}'.format(scaler_path))
    print('Elapsed time: {:.3f} s'.format(timer() - start_time))


def extract_features(data_config: str = 'configs/tnsse2021_salsa_feature_config.yml',
                     cond_num: float = 5,  # 5, 0
                     n_hopframes: int = 3,   # do not change
                     is_tracking: bool = True,  # Better to do tracking
                     is_compress_high_freq: bool = True,
                     task: str = 'feature_scaler') -> None:
    """
    Extract salsa features: log-linear spectrogram + normalized eigenvector (magnitude for FOA, phase for MIC)
    :param data_config: Path to data config file.
    :param cond_num: threshold for ddr for coherence test.
    :param n_hopframes: Number of adjacent frames to compute covariance matrix.
    :param is_tracking: If True, do noise-floor tracking.
    :param is_compress_high_freq: If True, compress high frequency region to reduce feature dimension.
    :param task: 'feature_scaler': extract feature and scaler, 'feature': only extract feature, 'scaler': only extract
        scaler.
    """
    # Load data config files
    with open(data_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    feature_type = 'salsa'

    # Parse config file
    audio_format = cfg['data']['format']
    fs = cfg['data']['fs']
    n_fft = cfg['data']['n_fft']
    hop_length = cfg['data']['hop_len']
    win_length = cfg['data']['win_len']

    # Doa info
    n_mics = 4
    fmin_doa = cfg['data']['fmin_doa']
    fmax_doa = cfg['data']['fmax_doa']
    fmax_doa = np.min((fmax_doa, fs // 2))
    n_bins = n_fft // 2 + 1
    lower_bin = np.int(np.floor(fmin_doa * n_fft / np.float(fs)))  # 512: 1; 256: 0
    upper_bin = np.int(np.floor(fmax_doa * n_fft / np.float(fs)))  # 9000Hz: 512: 192, 256: 96
    lower_bin = np.max((1, lower_bin))

    assert n_fft == 512 or n_fft == 256, 'only 256 or 512 fft is supported'
    if is_compress_high_freq:
        if n_fft == 512:
            freq_dim = 200
        elif n_fft == 256:
            freq_dim = 100
    else:
        freq_dim = n_fft // 2

    # Get feature descriptions
    feature_description = '{}fs_{}nfft_{}nhop_{}cond_{}fmaxdoa'.format(
        fs, n_fft, hop_length, int(cond_num), int(fmax_doa))
    if not is_tracking:
        feature_description = feature_description + '_notracking'
    if not is_compress_high_freq:
        feature_description = feature_description + '_nocompress'

    # Get feature extractor
    stft_feature_extractor = MagStftExtractor(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                              is_compress_high_freq=is_compress_high_freq)

    if audio_format == 'foa':
        splits = ['foa_dev', 'foa_eval']
    elif audio_format == 'mic':
        splits = ['mic_dev', 'mic_eval']
    else:
        raise ValueError('Unknown audio format {}'.format(audio_format))

    print('Feature description: {}'.format(feature_description))
    # Extract features
    if task in ['feature_scaler', 'feature']:
        for split in splits:
            print('============> Start extracting features for {} split'.format(split))
            start_time = timer()
            # Required directories
            audio_dir = os.path.join(cfg['data_dir'], split)
            feature_dir = os.path.join(cfg['feature_dir'], feature_type, audio_format, feature_description, split)
            # Empty feature folder
            shutil.rmtree(feature_dir, ignore_errors=True)
            os.makedirs(feature_dir, exist_ok=True)

            # Get audio list
            audio_fn_list = sorted(os.listdir(audio_dir))

            # Extract features
            for count, audio_fn in enumerate(tqdm(audio_fn_list)):
                full_audio_fn = os.path.join(audio_dir, audio_fn)
                audio_input, _ = librosa.load(full_audio_fn, sr=fs, mono=False, dtype=np.float32)
                # Extract stft feature (already remove the first frequency bin, correspond to fmin)
                stft_feature = stft_feature_extractor.extract(audio_input)  # (n_channels, n_timesteps, 200)

                # Extract mask and doa
                # Compute stft
                for imic in np.arange(n_mics):
                    stft = librosa.stft(y=np.asfortranarray(audio_input[imic, :]), n_fft=n_fft, hop_length=hop_length,
                                        center=True, window='hann', pad_mode='reflect')
                    if imic == 0:
                        n_frames = stft.shape[1]
                        afeature = np.zeros((n_bins, n_frames, n_mics), dtype='complex')
                    afeature[:, :, imic] = stft
                X = afeature[lower_bin:upper_bin, :, :]
                # compute normalized eigenvector
                normed_eigenvector_mat = extract_normalized_eigenvector(
                    X, condition_number=cond_num, n_hopframes=n_hopframes, is_tracking=is_tracking,
                    audio_format=audio_format, fs=fs, n_fft=n_fft, lower_bin=lower_bin,)

                # lower_bin now start at 0
                full_eigenvector_mat = np.zeros((n_mics - 1, n_frames, freq_dim))
                full_eigenvector_mat[:, :, :(upper_bin - lower_bin)] = np.transpose(normed_eigenvector_mat, (0, 2, 1))

                # Stack features
                audio_feature = np.concatenate((stft_feature, full_eigenvector_mat), axis=0)

                # Write features to file
                feature_fn = os.path.join(feature_dir, audio_fn.replace('wav', 'h5'))
                with h5py.File(feature_fn, 'w') as hf:
                    hf.create_dataset('feature', data=audio_feature, dtype=np.float32)
                tqdm.write('{}, {}, {}'.format(count, audio_fn, audio_feature.shape))

            print("Extracting feature finished! Elapsed time: {:.3f} s".format(timer() - start_time))

    # Compute feature mean and std for train set. For simplification, we use same mean and std for validation and
    # evaluation
    if task in ['feature_scaler', 'scaler']:
        feature_dir = os.path.join(cfg['feature_dir'], feature_type, audio_format, feature_description)
        compute_scaler(feature_dir=feature_dir, audio_format=audio_format)


if __name__ == '__main__':
    fire.Fire(extract_features)