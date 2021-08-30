"""
This module includes classes and functions to extract audio features and compute global mean and standard deviation of
the extracted features.
Reference: ff551c5 https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/
    master/utils/feature_extractor.py
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


class FeatureExtractor:
    """
    Base class for feature extraction.
    """
    def __init__(self, fs: int, n_fft: int, hop_length: int, n_mels: int, win_length: int = None, fmin: int = 50,
                 fmax: int = None, window: str = 'hann'):
        """
        :param fs: Sampling rate.
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param n_mels: Number of mel bands.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param fmin: Min frequency to extract feature (Hz).
        :param fmax: Max frequency to extract feature (Hz).
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
        self.melW = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        """
        raise NotImplementedError


class MelSpecGccExtractor(FeatureExtractor):
    """
    Extract log-mel spectrograms and GCC-PHAT features.
    """
    def __init__(self, fs: int, n_fft: int, hop_length: int, n_mels: int, win_length: int = None, fmin: int = 50,
                 fmax: int = None, window: str = 'hann'):
        """
        :param fs: Sampling rate.
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param n_mels: Number of mel bands.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param fmin: Min frequency to extract feature (Hz).
        :param fmax: Max frequency to extract feature (Hz).
        :param window: Type of window.
        """
        super().__init__(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, fmin=fmin,
                         fmax=fmax, window=window)
        self.n_mels = n_mels
        self.fs = fs

    def gcc_phat(self, sig, refsig) -> np.ndarray:
        """
        Compute GCC-PHAT between sig and refsig.
        :param sig: <np.ndarray: (n_samples,).
        :param refsig: <np.ndarray: (n_samples,).
        :return: gcc_phat: <np.ndarray: (1, n_frames, n_mels)>
        """
        ncorr = 2 * self.n_fft - 1
        n_fft = int(2 ** np.ceil(np.log2(np.abs(ncorr))))  # this n_fft double the length of win_length
        Px = librosa.stft(y=np.asfortranarray(sig),
                          n_fft=n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          center=True,
                          window=self.window,
                          pad_mode='reflect')
        Px_ref = librosa.stft(y=np.asfortranarray(refsig),
                              n_fft=n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              center=True,
                              window=self.window,
                              pad_mode='reflect')
        # Filter gcc spectrum, cutoff frequency = 4000Hz, buffer bandwidth = 400Hz
        freq_filter = np.ones((n_fft//2 + 1, 1))
        k_cutoff = int(4000 / self.fs * n_fft)
        k_buffer = int(400 / self.fs * n_fft)
        cos_x = np.arange(2 * k_buffer) * (np.pi/2) / (2 * k_buffer - 1)
        freq_filter[k_cutoff - k_buffer: k_cutoff + k_buffer, 0] = np.cos(cos_x)
        Px = Px * freq_filter
        Px_ref = Px_ref * freq_filter

        R = Px * np.conj(Px_ref)
        n_frames = R.shape[1]
        gcc_phat = []
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j * np.angle(spec)))
            cc = np.concatenate((cc[-self.n_mels // 2:], cc[:self.n_mels // 2]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        gcc_phat = gcc_phat[None, :, :]

        return gcc_phat

    def logmel(self, sig) -> np.ndarray:
        """
        Compute logmel of single channel signal
        :param sig: <np.ndarray: (n_samples,).
        :return: logmel: <np.ndarray: (1, n_frames, n_mels)>.
        """
        spec = np.abs(librosa.stft(y=np.asfortranarray(sig),
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length,
                                   center=True,
                                   window=self.window,
                                   pad_mode='reflect'))

        mel_spec = np.dot(self.melW, spec ** 2).T
        logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
        logmel_spec = np.expand_dims(logmel_spec, axis=0)

        return logmel_spec

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        :return: features <np.ndarray: (n_channel + n_channels*(n_channel-1)/2, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        features = []
        gcc_features = []
        for n in range(n_channels):
            features.append(self.logmel(audio_input[n]))
            for m in range(n + 1, n_channels):
                gcc_features.append(self.gcc_phat(sig=audio_input[m], refsig=audio_input[n]))

        features.extend(gcc_features)
        features = np.concatenate(features, axis=0)

        return features


class MelSpecIvExtractor(FeatureExtractor):
    """
    Extract Log-mel spectrograms and Intensity vector from FOA format.
    """
    def __init__(self, fs: int, n_fft: int, hop_length: int, n_mels: int, win_length: int = None, fmin: int = 50,
                 fmax: int = None, window: str = 'hann'):
        """
        :param fs: Sampling rate.
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param n_mels: Number of mel bands.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param fmin: Min frequency to extract feature (Hz).
        :param fmax: Max frequency to extract feature (Hz).
        :param window: Type of window.
        """
        super().__init__(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, fmin=fmin,
                         fmax=fmax, window=window)
        self.eps = 1e-8

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (4, n_samples)>.
        :return: logmel <np.ndarray: (7, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        features = []
        X = []

        for i_channel in range(n_channels):
            spec = librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                center=True,
                                window=self.window,
                                pad_mode='reflect')
            X.append(np.expand_dims(spec, axis=0))  # 1 x n_bins x n_frames

            # compute logmel
            mel_spec = np.dot(self.melW, np.abs(spec) ** 2).T
            logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
            logmel_spec = np.expand_dims(logmel_spec, axis=0)
            features.append(logmel_spec)

        # compute intensity vector: for ambisonic signal, n_channels = 4
        X = np.concatenate(X, axis=0)  # 4 x n_bins x n_frames
        IVx = np.real(np.conj(X[0, :, :]) * X[1, :, :])
        IVy = np.real(np.conj(X[0, :, :]) * X[2, :, :])
        IVz = np.real(np.conj(X[0, :, :]) * X[3, :, :])

        normal = np.sqrt(IVx ** 2 + IVy ** 2 + IVz ** 2) + self.eps
        IVx = np.dot(self.melW, IVx / normal).T  # n_frames x n_mels
        IVy = np.dot(self.melW, IVy / normal).T
        IVz = np.dot(self.melW, IVz / normal).T

        # add intensity vector to logmel
        features.append(np.expand_dims(IVx, axis=0))
        features.append(np.expand_dims(IVy, axis=0))
        features.append(np.expand_dims(IVz, axis=0))
        feature = np.concatenate(features, axis=0)

        return feature


class MelSpecExtractor(FeatureExtractor):
    """
    Extract single-channel or multi-channel logmel spectrograms.
    """
    def __init__(self, fs: int, n_fft: int, hop_length: int, n_mels: int, win_length: int = None, fmin: int = 50,
                 fmax: int = None, window: str = 'hann'):
        """
        :param fs: Sampling rate.
        :param n_fft: Number of FFT points.
        :param hop_length: Number of sample for hopping.
        :param n_mels: Number of mel bands.
        :param win_length: Window length <= n_fft. If None, assign n_fft
        :param fmin: Min frequency to extract feature (Hz).
        :param fmax: Max frequency to extract feature (Hz).
        :param window: Type of window.
        """
        super().__init__(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, fmin=fmin,
                         fmax=fmax, window=window)

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        :return: logmel_features <np.ndarray: (n_channels, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        logmel_features = []

        for i_channel in range(n_channels):
            spec = np.abs(librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                                       n_fft=self.n_fft,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length,
                                       center=True,
                                       window=self.window,
                                       pad_mode='reflect'))

            mel_spec = np.dot(self.melW, spec**2).T
            logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
            logmel_spec = np.expand_dims(logmel_spec, axis=0)
            logmel_features.append(logmel_spec)

        logmel_features = np.concatenate(logmel_features, axis=0)

        return logmel_features


class LinSpecIvExtractor:
    """
    Extract single-channel or multi-channel log-linear spectrogram + intensity vector.
    Return feature of shape 4 x n_timesteps x 200
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
        self.eps = 1e-8
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
        :param audio_input: <np.ndarray: (4, n_samples)>.
        :return: features <np.ndarray: (7, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        features = []

        X  = []
        for i_channel in range(n_channels):
            spec = librosa.stft(y=np.asfortranarray(audio_input[i_channel]),
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                center=True,
                                window=self.window,
                                pad_mode='reflect')

            X.append(np.expand_dims(spec, axis=0))  # 1 x n_bins x n_frames

            spec = np.dot(self.W, np.abs(spec)**2).T
            log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
            log_spec = np.expand_dims(log_spec, axis=0)
            features.append(log_spec)

        # compute intensity vector: for ambisonic signal, n_channels = 4
        X = np.concatenate(X, axis=0)  # 4 x n_bins x n_frames
        IVx = np.real(np.conj(X[0, :, :]) * X[1, :, :])
        IVy = np.real(np.conj(X[0, :, :]) * X[2, :, :])
        IVz = np.real(np.conj(X[0, :, :]) * X[3, :, :])

        normal = np.sqrt(IVx ** 2 + IVy ** 2 + IVz ** 2) + self.eps
        IVx = np.dot(self.W, IVx / normal).T  # n_frames x n_freqs
        IVy = np.dot(self.W, IVy / normal).T
        IVz = np.dot(self.W, IVz / normal).T

        # add intensity vector to logmel
        features.append(np.expand_dims(IVx, axis=0))
        features.append(np.expand_dims(IVy, axis=0))
        features.append(np.expand_dims(IVz, axis=0))
        features = np.concatenate(features, axis=0)

        return features


class LogSpecGccExtractor:
    """
    Extract single-channel or multi-channel log-linear spectrogram + GCC-PHAT.
    Return feature of shape M + M(M-1)/2 x n_timesteps x 200
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
                self.n_freqs = 200
                self.W = np.zeros((200, 257), dtype=np.float32)
                for i in np.arange(192):
                    self.W[i, i+1] = 1.0
                for i in np.arange(192, 200):
                    if i < 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 8] = 1/8
                    elif i == 199:
                        self.W[i, 193 + (i-192) * 8: 193 + (i-192) * 8 + 7] = 1/8
            elif n_fft == 256:
                self.n_freqs = 100
                self.W = np.zeros((100, 129), dtype=np.float32)
                for i in np.arange(96):
                    self.W[i, i+1] = 1.0
                for i in np.arange(96, 100):
                    if i < 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 8] = 1/8
                    elif i == 99:
                        self.W[i, 97 + (i-96) * 8: 97 + (i-96) * 8 + 7] = 1/8
        else:
            self.n_freqs = n_fft // 2
            self.W = np.zeros((n_fft // 2, n_fft // 2 + 1), dtype=np.float32)
            for i in np.arange(n_fft // 2):
                self.W[i, i + 1] = 1.0

    def gcc_phat(self, sig, refsig) -> np.ndarray:
        """
        Compute GCC-PHAT between sig and refsig.
        :param sig: <np.ndarray: (n_samples,).
        :param refsig: <np.ndarray: (n_samples,).
        :return: gcc_phat: <np.ndarray: (1, n_frames, n_freqs)>
        """
        ncorr = 2 * self.n_fft - 1
        n_fft = int(2 ** np.ceil(np.log2(np.abs(ncorr))))  # this n_fft double the length of win_length
        Px = librosa.stft(y=np.asfortranarray(sig),
                          n_fft=n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          center=True,
                          window=self.window,
                          pad_mode='reflect')
        Px_ref = librosa.stft(y=np.asfortranarray(refsig),
                              n_fft=n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              center=True,
                              window=self.window,
                              pad_mode='reflect')
        R = Px * np.conj(Px_ref)
        n_frames = R.shape[1]
        gcc_phat = []
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j * np.angle(spec)))
            cc = np.concatenate((cc[-self.n_freqs // 2:], cc[:self.n_freqs // 2]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        gcc_phat = gcc_phat[None, :, :]

        return gcc_phat

    def logspec(self, sig) -> np.ndarray:
        """
        Compute log power spectrogram of single channel signal
        :param sig: <np.ndarray: (n_samples,).
        :return: logspec: <np.ndarray: (1, n_frames, n_freqs)>.
        """
        spec = np.abs(librosa.stft(y=np.asfortranarray(sig),
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length,
                                   center=True,
                                   window=self.window,
                                   pad_mode='reflect'))

        spec = np.dot(self.W, spec ** 2).T
        logspec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)
        logspec = np.expand_dims(logspec, axis=0)

        return logspec

    def extract(self, audio_input: np.ndarray) -> np.ndarray:
        """
        :param audio_input: <np.ndarray: (n_channels, n_samples)>.
        :return: features <np.ndarray: (n_channels + n_pairs, n_timeframes, n_features)>.
        """
        n_channels = audio_input.shape[0]
        features = []
        gcc_features = []
        for n in range(n_channels):
            features.append(self.logspec(audio_input[n]))
            for m in range(n + 1, n_channels):
                gcc_features.append(self.gcc_phat(sig=audio_input[m], refsig=audio_input[n]))

        features.extend(gcc_features)
        features = np.concatenate(features, axis=0)

        return features


def select_extractor(feature_type: str, fs: int, n_fft: int, hop_length: int, n_mels: int, win_length: int = None,
                     fmin: int = 50, fmax: int = None)\
        -> None:
    """
    Select feature extractor based on feature_type.
    :param feature_type: Choices are:
        'melspec': log-mel spectrograms.
        'melspecgcc': log-mel spectrograms + gcc.
        'melspeciv': log-mel spectrograms + iv
        'linspeciv': log-linear spectrograms + iv
        'linspecgcc': log-liner spectrograms + gcc
    :param fs: Sampling rate.
    :param n_fft: Number of FFT points.
    :param hop_length: Number of sample for hopping.
    :param n_mels: Number of mel bands.
    :param win_length: Window length.
    :param fmin: Min frequency to extract feature (Hz).
    :param fmax: Max frequency to extract feature (Hz).
    """
    if feature_type == 'melspec':
        extractor = MelSpecExtractor(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length,
                                     fmin=fmin, fmax=fmax)
    elif feature_type == 'melspeciv':
        extractor = MelSpecIvExtractor(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length,
                                       fmin=fmin, fmax=fmax)
    elif feature_type == 'melspecgcc':
        extractor = MelSpecGccExtractor(fs=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length,
                                        fmin=fmin, fmax=fmax)
    elif feature_type == 'linspeciv':
        extractor = LinSpecIvExtractor(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                       is_compress_high_freq=(n_mels < n_fft // 2))
    elif feature_type == 'linspecgcc':
        extractor = LogSpecGccExtractor(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                        is_compress_high_freq=(n_mels < n_fft // 2))
    else:
        raise NotImplementedError('Feature type {} is not implemented!'.format(feature_type))

    return extractor


def compute_scaler(feature_dir: str, audio_format: str) -> None:
    """
    Compute feature mean and std vectors for normalization.
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
    scaler_dict = {}
    scalar_scaler_dict = {}
    for i_chan in range(n_channels):
        scaler_dict[i_chan] = preprocessing.StandardScaler()
        scalar_scaler_dict[i_chan] = preprocessing.StandardScaler()

    # Iterate through data
    for count, feature_fn in enumerate(tqdm(feature_fn_list)):
        full_feature_fn = os.path.join(train_feature_dir, feature_fn)
        with h5py.File(full_feature_fn, 'r') as hf:
            afeature = hf['feature'][:]  # (n_chanels, n_timesteps, n_features)
            for i_chan in range(n_channels):
                scaler_dict[i_chan].partial_fit(afeature[i_chan, :, :])  # (n_timesteps, n_features)
                scalar_scaler_dict[i_chan].partial_fit(np.reshape(afeature[i_chan, :, :], (-1, 1)))  # (n_timesteps * n_features, 1)

    # Extract mean and std
    feature_mean = []
    feature_std = []
    feature_mean_scalar = []
    feature_std_scalar = []
    for i_chan in range(n_channels):
        feature_mean.append(scaler_dict[i_chan].mean_)
        feature_std.append(np.sqrt(scaler_dict[i_chan].var_))
        feature_mean_scalar.append(scalar_scaler_dict[i_chan].mean_)
        feature_std_scalar.append(np.sqrt(scalar_scaler_dict[i_chan].var_))

    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)
    feature_mean_scalar = np.array(feature_mean_scalar)
    feature_std_scalar = np.array(feature_std_scalar)

    # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
    feature_mean = np.expand_dims(feature_mean, axis=1)
    feature_std = np.expand_dims(feature_std, axis=1)
    feature_mean_scalar = np.expand_dims(feature_mean_scalar, axis=1)
    feature_std_scalar = np.expand_dims(feature_std_scalar, axis=1)

    scaler_path = os.path.join(feature_dir, audio_format + '_feature_scaler.h5')
    with h5py.File(scaler_path, 'w') as hf:
        hf.create_dataset('mean', data=feature_mean, dtype=np.float32)
        hf.create_dataset('std', data=feature_std, dtype=np.float32)
        hf.create_dataset('scalar_mean', data=feature_mean_scalar, dtype=np.float32)
        hf.create_dataset('scalar_std', data=feature_std_scalar, dtype=np.float32)

    print('Features shape: {}'.format(afeature.shape))
    print('mean {}: {}'.format(feature_mean.shape, feature_mean))
    print('std {}: {}'.format(feature_std.shape, feature_std))
    print('scalar mean {}: {}'.format(feature_mean_scalar.shape, feature_mean_scalar))
    print('scalar std {}: {}'.format(feature_std_scalar.shape, feature_std_scalar))
    print('Scaler path: {}'.format(scaler_path))
    print('Elapsed time: {:.3f} s'.format(timer() - start_time))


def extract_features(data_config: str = 'configs/tnsse2021_feature_config.yml', feature_type: str = 'linspeciv',
                     task: str = 'feature_scaler', is_compressed_freq: bool = True) -> None:
    """
    Extract features
    :param data_config: Path to data config file.
    :param feature_type: Choices are:
        'melspec': log-mel spectrograms.
        'melspecgcc': log-mel spectrograms + gcc.
        'melspeciv': log-mel spectrograms + iv
        'linspeciv': log-linear spectrograms + iv
        'linspecgcc': log-liner spectrograms + gcc
    :param task: 'feature_scaler': extract feature and scaler, 'feature': only extract feature, 'scaler': only extract
        scaler.
    :param is_compressed_freq: if feature type is 'linspeciv' or 'linspecigcc', if is_compressed_freq=True, the high
        frequency bands above 9000Hz will be compressed by a factor of 8
    """
    # Load data config files
    with open(data_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Parse config file
    audio_format = cfg['data']['format']
    fs = cfg['data']['fs']
    n_fft = cfg['data']['n_fft']
    hop_length = cfg['data']['hop_len']
    win_length = cfg['data']['win_len']
    fmin = cfg['data']['fmin']
    fmax = cfg['data']['fmax']
    n_mels = cfg['data']['n_mels']
    fmax = np.min((fmax, fs//2))

    if feature_type in ['linspeciv', 'linspecgcc']:
        assert n_fft == 256 or n_fft == 512, 'nfft = {} is not supported for {}'.format(n_fft, feature_type)
        if is_compressed_freq:
            if n_fft == 512:
                n_mels = 200
            elif n_fft == 256:
                n_mels = 100
        else:
            n_mels = n_fft // 2

    # Get feature descriptions
    if feature_type in ['linspeciv', 'linspecgcc']:
        feature_description = '{}fs_{}nfft_{}nhop_{}nfreqs'.format(fs, n_fft, hop_length, n_mels)
    else:
        feature_description = '{}fs_{}nfft_{}nhop_{}nmels'.format(fs, n_fft, hop_length, n_mels)

    print('Feature description: {}'.format(feature_description))

    # Get feature extractor
    feature_extractor = select_extractor(feature_type=feature_type, fs=fs, n_fft=n_fft, hop_length=hop_length,
                                         n_mels=n_mels, win_length=win_length, fmin=fmin, fmax=fmax)

    if audio_format == 'foa':
        splits = ['foa_dev', 'foa_eval']
    elif audio_format == 'mic':
        splits = ['mic_dev', 'mic_eval']
    else:
        raise ValueError('Unknown audio format {}'.format(audio_format))

    # Extract features
    if task in ['feature_scaler', 'feature']:
        for split in splits:
            print('============> Start extracting features {} for {} split'.format(feature_type, split))
            start_time = timer()
            # Required directories
            audio_dir = os.path.join(cfg['data_dir'], split)
            feature_dir = os.path.join(cfg['feature_dir'], feature_type, feature_description, split)
            # Empty feature folder
            shutil.rmtree(feature_dir, ignore_errors=True)
            os.makedirs(feature_dir, exist_ok=True)

            # Get audio list
            audio_fn_list = sorted(os.listdir(audio_dir))

            # Extract features
            for count, audio_fn in enumerate(tqdm(audio_fn_list)):
                full_audio_fn = os.path.join(audio_dir, audio_fn)
                audio_input, _ = librosa.load(full_audio_fn, sr=fs, mono=False, dtype=np.float32)
                audio_feature = feature_extractor.extract(audio_input)  # (n_channels, n_timesteps, n_mels)

                # Write features to file
                feature_fn = os.path.join(feature_dir, audio_fn.replace('wav', 'h5'))
                with h5py.File(feature_fn, 'w') as hf:
                    hf.create_dataset('feature', data=audio_feature, dtype=np.float32)
                tqdm.write('{}, {}, {}'.format(count, audio_fn, audio_feature.shape))

            print("Extracting feature finished! Elapsed time: {:.3f} s".format(timer() - start_time))

    # Compute feature mean and std for train set. For simplification, we use same mean and std for validation and
    # evaluation
    if task in ['feature_scaler', 'scaler']:
        feature_dir = os.path.join(cfg['feature_dir'], feature_type, feature_description)
        compute_scaler(feature_dir=feature_dir, audio_format=audio_format)


if __name__ == '__main__':
    fire.Fire(extract_features)
