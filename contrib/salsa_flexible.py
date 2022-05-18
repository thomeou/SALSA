#!/usr/env/bin python
# -*- coding:utf-8 -*-


"""
This module is a reimplementation of the SALSA and SALSA-Lite features,
refactored to work on-the-fly (CPU) and with an arbitrary no. of microphones.

References:
* github.com/thomeou/SALSA/blob/master/dataset/salsa_lite_feature_extraction.py
* https://gist.github.com/andres-fr/d923e1df7de4dd6e0af34b28a2a7ef04
* https://github.com/thomeou/SALSA/issues/4


Usage: simply instantiate the SalsaFeatures (for SALSA) or SalsaLitefeatures
(for SALSA-Lite) classes. Code examples can be found in the respective class
docstrings.

--------------------------------------------------------------------------------

MIT License below is compatible with the original SALSA repository:

Copyright 2022 aferro (OrcID: 0000-0003-3830-3595 )

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import librosa
import numpy as np


# #############################################################################
# # HELPERS
# #############################################################################
def stacked_covmat_eigh(arr):
    """
    Given an array of shape hape ``(freqbins, t, ch)``, first computes
    an array of shape ``(freqbins, ch, ch)``, where for each freqbin
    we compute the ``(ch, ch)`` spatial covariance matrix, averaged
    among all given ``t``. Then, it computes the eigendecomposition
    of the covariance matrices.

    :returns: The pair ``(ews, evs)`` of shapes ``(freqbins, ch)``
      and ``(freqbins, ch, ch)``, containing the per-freqbin eigenvalues
      and eigenvectors, respectively.

    This function makes use of two main speedup strategies: first, all
    freqbins are computed in parallel. Second, since the covmat is
    Hermitian, we only need one of its triangles to compute the
    decomposition, via ``np.linalg.eigh(covmats, UPLO='U')``.
    """
    f, _, ch = arr.shape
    covmats = np.zeros((f, ch, ch), dtype=arr.dtype)
    for i in range(ch):
        for j in range(i, ch):
            ch_i, ch_j = arr[:, :, i], arr[:, :, j]
            covmats[:, i, j] = (ch_i * ch_j.conj()).sum(axis=-1)
    ews, evs = np.linalg.eigh(covmats, UPLO="U")
    #
    return ews, evs


class SalsaNoiseFloorTracker:
    """
    Heuristic noise floor tracker as proposed and implemented in the SALSA
    paper, and refactored here into a class.
    """
    def __init__(self, initial_floor, steps=3,
                 up_ratio_initial=1.02, up_ratio_many=1.002, down_ratio=0.98,
                 epsilon=1e-6):
        """
        This noise floor tracker operates on each frequency bin independently,
        through time. Depending on the values, the floor will rise or sink.

        :param initial_floor: Array of shape ``(freq,)``, representing the
          initial state of the noise floor tracker.
        :param int steps: Number of consecutive steps to be considered in
          order to decide how to update the noise floor.
        :param up_ratio_initial: Ratio to raise noise floor within the initial
          ``steps``.
        :param float up_ratio_many: Slower ratio to raise noise floor when the
          number of consecutive ``steps`` has been surpassed.
        :param float down_ratio: Ratio to lower noise floor.
        :param float epsilon: Lower bound for the floor values, will be clipped
          to this.
        """
        self.steps = steps
        self.epsilon = epsilon
        self.up_init = up_ratio_initial
        self.up_many = up_ratio_many
        self.down = down_ratio
        #
        self.floor = initial_floor.copy()
        if (self.floor < epsilon).any():
            print(f"WARNING: modifying floor values to be >= {epsilon}.")
            self.floor[self.floor < epsilon] = epsilon
        #
        self.tracker = np.zeros(self.floor.shape, dtype=np.int64)

    def __call__(self, frame, floor_mask_ratio=1.5):
        """
        Call this method with a new frame of frequency bins to update the
        current noise floor and retrieve a mask with the bins that are
        considered above noise floor.

        :param frame: Array of same shape as ``initial_floor``.
        :param floor_mask_ratio: All bins with values above the updated
          noise floor times this ratio will be True in the returned mask.
        :returns: Mask of same shape as given frame, with True whenever
          the entry is above noise floor times ``floor_mask_ratio``.
        """
        # check "above noise floor" entries and update tracker
        above_floor = frame > self.floor
        not_above_floor = ~above_floor
        self.tracker += above_floor
        few_consecutive = above_floor & (self.tracker <= self.steps)
        many_consecutive = above_floor & (self.tracker > self.steps)
        # floor rises more for the first consecutive samples above noise. After
        # several consecutive samples, floor rises more slowly
        self.floor[few_consecutive] *= self.up_init
        self.floor[many_consecutive] *= self.up_many
        # floor sinks whenever signal is not above noise, but stays >=epsilon
        self.floor[not_above_floor] *= self.down
        self.floor[self.floor < self.epsilon] = self.epsilon
        # Reset tracker for any reading that wasn't above noise floor
        self.tracker[not_above_floor] = 0
        # Compute and return above-noise-floor mask
        mask = frame > (floor_mask_ratio * self.floor)
        return mask


class SpatialFeaturesAbstract:
    """
    This class contains all the common functionality among different SALSA
    representations:
    * Calculation of lower, upper and cutoff frequencies
    * Calculation of frequency normalization vector
    * Method to calculate STFTs and log-mel spectrograms
    * Full feature pipeline that computes (and optionally clips) the STFTs and
      log-mels, computes (and optionally clips) the spatial features,
      and finally retrieves the log-mels and spatial features concatenated.

    To use it, extend the ``features(stft, norm_freq, **kwargs):`` method
    with the desired feature and then call the instance with the desired
    kwargs. See SALSA and SALSA-Lite examples below.
    """

    SOUND_SPEED = 343  # m/s
    F_DTYPE = np.float32

    def __init__(self, fs=24000, stft_winsize=512, hop_length=300,
                 fmin_doa=50, fmax_doa=2000, fmax_spec=9000):
        """
        """
        n_bins = stft_winsize // 2 + 1
        # freqs can be cropped between lower and cutoff bin to prevent spatial
        # aliasing. Once cropped, all phase feats above upper can be set to 0
        lower_bin = np.int(np.floor(fmin_doa * stft_winsize / np.float(fs)))  # 512: 1; 256: 0
        lower_bin = np.max((1, lower_bin))
        upper_bin = np.int(np.floor(fmax_doa * stft_winsize / np.float(fs)))  # 9000Hz: 512: 192, 256: 96
        # Cutoff frequency for spectrograms
        cutoff_bin = np.int(np.floor(fmax_spec * stft_winsize / np.float(fs)))  # 9000 Hz, 512 nfft: cutoff_bin = 192
        assert upper_bin <= cutoff_bin, "Upper bin for spatial feature is " + \
            "higher than cutoff bin for spectrogram!"
        # Normalization factor
        self.delta = 2 * np.pi * fs / (stft_winsize * self.SOUND_SPEED)
        # feature bins will be divided by this: (freq, 1)
        self.norm_freq = np.arange(n_bins, dtype=self.F_DTYPE)[:, None]
        self.norm_freq[0, 0] = 1  # from salsa lite code
        self.norm_freq *= self.delta
        #
        self.stft_winsize = stft_winsize
        self.hop_length = hop_length
        self.n_bins = n_bins
        #
        self.lobin, self.upbin, self.cutbin = lower_bin, upper_bin, cutoff_bin

    def spectrograms(self, wavchans):
        """
        :param wavchans: Float array of shape ``(channels, samples)``
        :returns: A pair ``(stfts, log_specs)``, each element of shape
          ``(channels, freqbins, time)``.
        """
        n_chans, _ = wavchans.shape  # (n_chans, n_samples)
        # first compute logmel spectrograms for all channels
        log_specs = []
        for ch_i in np.arange(n_chans):
            stft = librosa.stft(
                y=np.asfortranarray(wavchans[ch_i, :]),
                n_fft=self.stft_winsize, hop_length=self.hop_length,
                center=True, window="hann", pad_mode="reflect")
            if ch_i == 0:
                n_frames = stft.shape[1]
                stfts = np.zeros((n_chans, self.n_bins, n_frames),
                                 dtype="complex")
            stfts[ch_i, :, :] = stft
            # Compute log linear power spectrum
            spec = (np.abs(stft) ** 2)
            log_spec = librosa.power_to_db(
                spec, ref=1.0, amin=1e-10, top_db=None)
            log_specs.append(log_spec)
        log_specs = np.stack(log_specs)  # (ch, freqbins, time)
        #
        return stfts, log_specs

    def features(self, stft, norm_freq, **kwargs):
        """
        Extend this method with the desired functionality. It must fulfill
        the following interface:

        * Inputs: ``(stft, norm_freq, **kwargs)``, where stft is a complex
          array of shape ``(ch, freq, t)``, and norm_freq is a float array
          of shape ``(freq, 1)``.
        * Output: Feature array of shape ``(ch, freq, t)``

        See e.g. the ``SalsaFeatures`` and ``SalsaLiteFeatures`` classes.
        """
        raise NotImplementedError("Implement features here!")

    def __call__(self, wavchans, clip_freqs, clip_spatial_alias,
                 **feat_kwargs):
        """
        :param wavchans: Float array of shape ``(channels, samples)``
        :param bool clip_freqs: Whether to remove undesired frequency bins
        :param bool clip_spatial_alias: Whether to zero-out potentially
          aliasing freqbins from the spatial features
        :returns: Array of shape ``(n_feats, freqbins, time)``, where the
          number of features equals ``channels + spatial_feats``, because it
          is a concatenation of the logmel spectrograms and the result of the
          ``features`` method.
        """
        _, _ = wavchans.shape  # (n_chans, n_samples) test if rank 2
        assert wavchans.dtype == self.F_DTYPE, f"{self.F_DTYPE} expected!"
        stfts, log_specs = self.spectrograms(wavchans)  # (ch, f, t)

        if clip_freqs:
            stfts = stfts[:, self.lobin:self.cutbin]
            log_specs = log_specs[:, self.lobin:self.cutbin]
            norm_freq = self.norm_freq[self.lobin:self.cutbin, :]
        else:
            norm_freq = self.norm_freq

        spatial_feats = self.features(stfts, norm_freq, **feat_kwargs)

        if clip_spatial_alias:
            spatial_feats[:, self.upbin:] = 0
        result = np.concatenate([log_specs, spatial_feats])
        return result


# #############################################################################
# # SALSA
# #############################################################################
class SalsaFeatures(SpatialFeaturesAbstract):
    """
    On-the-fly, parallelized CPU implementation of the SALSA features from
    the original paper.

    Usage example::

      sf = SalsaFeatures(fs=sr, stft_winsize=STFT_WINSIZE,
                         hop_length=STFT_HOP,
                         fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
      s = sf(wav, clip_freqs=True, clip_spatial_alias=False,
             ew_thresh=5.0, covmat_avg_neighbours=3,
             is_tracking=True, floor_mask_ratio=1.5)
    """

    def features(self, stfts,
                 norm_freq,
                 ew_thresh: float = 5.0,
                 covmat_avg_neighbours: int = 3,
                 is_tracking: bool = True,
                 floor_mask_ratio: float = 1.5):
        """
        This is a parallelized version of extract_normalized_eigenvector as
        originally implemented. This version has been tested to be correct up
        to sign flip in eigenvectors (since eigendecomposition is invariant to
        eigenvector sign). See class docstring for usage example.

        :param stfts: complex STFT of shape ``(n_chans, n_bins, n_frames)``,
          clipped between lower_bin and upper_bin.
        :param norm_freq: Array of shape ``(n_bins, 1)``, used to normalize
          features by frequency as explained in the paper.
        :param float ew_thresh: Required ratio between largest and 2nd-largest
          eigenvalue, used in the coherence test: all timefreq bins with
          covmats below this ratio will be ignored.
        :param int covmat_avg_neighbours: At each timepoint, the function will
          include this many points to the left and right to calculate the avg
          spatial covariance matrix. E.g. if 3 is given, 7 neighbouring
          matrices in total will be averaged.
        :param is_tracking: If True, use a heuristic noise-floor tracker to
          ignore noisy freqbins.
        :param float floor_mask_ratio: Any timefreq bins with intensity below
          noise level times this float will be considered noisy and ignored.
        :returns: Array of shape ``(n_chans-1, n_bins, n_frames)`` containing
          the SALSA features.
        """
        stfts = stfts.transpose(1, 2, 0)  # (freq, t, ch)
        n_bins, n_frames, n_chans = stfts.shape
        result = np.zeros((n_chans - 1, n_bins, n_frames))

        # padding stfts for avg covmat computation
        stft_pad = np.pad(
            stfts, ((0, 0), (covmat_avg_neighbours, covmat_avg_neighbours),
                    (0, 0)), "wrap")

        # amplitude spectrogram
        signal_magspec = np.abs(  # (freqs, T)
            stft_pad[:, covmat_avg_neighbours:covmat_avg_neighbours + n_frames,
                     0])

        # Initial noisefloor assuming first few frames are noise
        noise_floor = 0.5 * np.mean(signal_magspec[:, 0:5], axis=1)  # (freqs,)
        noise_tracker = SalsaNoiseFloorTracker(
            initial_floor=noise_floor, steps=3,
            up_ratio_initial=1.02, up_ratio_many=1.002,
            down_ratio=0.98, epsilon=1e-6)

        # Default mask is always all ones, so define it just once
        if not is_tracking:
            allpass_mask = np.ones(signal_magspec.shape[0], dtype=np.bool)

        for iframe, magspec_col in enumerate(signal_magspec.T,
                                             covmat_avg_neighbours):
            # Optionally, use noise tracker to mask out noisy bins
            if is_tracking:
                mask = noise_tracker(magspec_col,
                                     floor_mask_ratio=floor_mask_ratio)
            else:
                mask = allpass_mask

            # Compute spatial covmat eigendecomposition for all non-noisy bins
            readings = stft_pad[mask, iframe - covmat_avg_neighbours:
                                iframe + covmat_avg_neighbours + 1, :]
            ews, evs = stacked_covmat_eigh(readings)

            # Further remove from mask any bins with bad coherence
            good_coherence_mask = ews[:, -1] > (ews[:, -2] * ew_thresh)
            mask[mask] = good_coherence_mask

            # compute SALSA features for any non-masked bins
            evs = evs[good_coherence_mask]
            max_evs = evs[:, :, -1]  # all "last columns"
            norm_evs = np.angle(max_evs[:, 0:1].conj() * max_evs[:, 1:])
            norm_evs /= norm_freq[mask]
            # update result
            result[:, mask, iframe - covmat_avg_neighbours] = norm_evs.T
        #
        return result  # (ch, freq, t)


# #############################################################################
# # SALSA LITE
# #############################################################################
class SalsaLiteFeatures(SpatialFeaturesAbstract):
    """
    On-the-fly, parallelized CPU implementation of the SALSA-Lite features from
    the original paper.

    Usage example::

    slf = SalsaLiteFeatures(fs=sr, stft_winsize=STFT_WINSIZE,
                            hop_length=STFT_HOP,
                            fmin_doa=50, fmax_doa=2000, fmax_spec=9000)
    sl = slf(wav, clip_freqs=True, clip_spatial_alias=False)
    """

    def features(self, stfts, norm_freq):
        """
        This is a parallelized version of extract_normalized_eigenvector as
        originally implemented. This version has been tested to be correct up
        to sign flip in eigenvectors (since eigendecomposition is invariant to
        eigenvector sign). See class docstring for usage example.

        :param stfts: complex STFT of shape ``(n_chans, n_bins, n_frames)``,
          clipped between lower_bin and upper_bin.
        :param norm_freq: Array of shape ``(n_bins, 1)``, used to normalize
          features by frequency as explained in the paper.
        """
        result = np.angle(stfts[None, 0].conj() * stfts[1:])
        result /= norm_freq
        return result  # (ch, freq, t)
