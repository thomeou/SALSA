"""
@ Tho Nguyen, NTU, 2021 04 07
This module includes code to do data augmentation in STFT domain on numpy array:
    1. random volume
    2. random cutout
    3. spec augment
    4. freq shift
==================================================
Example how to use data augmentation
# import
from transforms import CompositeCutout, ComposeTransformNp, RandomShiftUpDownNp, RandomVolumeNp
# call transform
train_transform = ComposeTransformNp([
    RandomShiftUpDownNp(freq_shift_range=10),
    RandomVolumeNp(),
    CompositeCutout(image_aspect_ratio=320 / 128),  # 320: number of frames, 128: n_mels
    ])
# perform data augmentation
X = train_transform(X)  # X size: 1 x n_frames x n_mels
"""
import numpy as np


class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x


class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError


class RandomCutoutNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features) or (n_time_steps, n_features)>: input spectrogram.
        :return: random cutout x
        """
        # get image size
        image_dim = x.ndim
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        # Initialize output
        output_img = x.copy()
        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        if image_dim == 2:
            output_img[top:top + h, left:left + w] = c
        else:
            if self.n_zero_channels is None:
                output_img[:, top:top + h, left:left + w] = c
            else:
                output_img[:-self.n_zero_channels,  top:top + h, left:left + w] = c
                if self.is_filled_last_channels:
                    output_img[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return output_img


class SpecAugmentNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[1]
        n_freqs = x.shape[2]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, start_idx:start_idx + dur, :] = random_value
            else:
                new_spec[:-self.n_zero_channels, start_idx:start_idx + dur, :] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, start_idx:start_idx + dur, :] = 0.0

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            if self.n_zero_channels is None:
                new_spec[:, :, start_idx:start_idx + dur] = random_value
            else:
                new_spec[:-self.n_zero_channels, :, start_idx:start_idx + dur] = random_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, :, start_idx:start_idx + dur] = 0.0

        return new_spec


class RandomCutoutHoleNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 8,
                 max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                 is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
        n_cutout_holes = self.n_max_holes
        for ihole in np.arange(n_cutout_holes):
            # w = np.random.randint(4, self.max_w_size, 1)[0]
            # h = np.random.randint(4, self.max_h_size, 1)[0]
            w = self.max_w_size
            h = self.max_h_size
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.filled_value is None:
                filled_value = np.random.uniform(min_value, max_value)
            else:
                filled_value = self.filled_value
            if self.n_zero_channels is None:
                new_spec[:, top:top + h, left:left + w] = filled_value
            else:
                new_spec[:-self.n_zero_channels, top:top + h, left:left + w] = filled_value
                if self.is_filled_last_channels:
                    new_spec[-self.n_zero_channels:, top:top + h, left:left + w] = 0.0

        return new_spec


class CompositeCutout(DataAugmentNumpyBase):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio,
                                            n_zero_channels=n_zero_channels,
                                            is_filled_last_channels=is_filled_last_channels)
        self.spec_augment = SpecAugmentNp(always_apply=True, n_zero_channels=n_zero_channels,
                                          is_filled_last_channels=is_filled_last_channels)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True, n_zero_channels=n_zero_channels,
                                                     is_filled_last_channels=is_filled_last_channels)

    def apply(self, x: np.ndarray):
        choice = np.random.randint(0, 3, 1)[0]
        if choice == 0:
            return self.random_cutout(x)
        elif choice == 1:
            return self.spec_augment(x)
        elif choice == 2:
            return self.random_cutout_hole(x)


class RandomShiftUpDownNp(DataAugmentNumpyBase):
    """
    This data augmentation random shift the spectrogram up or down.
    """
    def __init__(self, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect',
                 n_last_channels: int = 0):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels

    def apply(self, x: np.ndarray):
        n_channels, n_timesteps, n_features = x.shape
        if self.freq_shift_range is None:
            self.freq_shift_range = int(n_features * 0.08)
        shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
        if self.direction is None:
            direction = np.random.choice(['up', 'down'], 1)[0]
        else:
            direction = self.direction
        new_spec = x.copy()
        if self.n_last_channels == 0:
            if direction == 'up':
                new_spec = np.pad(new_spec, ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
            else:
                new_spec = np.pad(new_spec, ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
        else:
            if direction == 'up':
                new_spec[:-self.n_last_channels] = np.pad(
                    new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (shift_len, 0)), mode=self.mode)[:, :, 0:n_features]
            else:
                new_spec[:-self.n_last_channels] = np.pad(
                    new_spec[:-self.n_last_channels], ((0, 0), (0, 0), (0, shift_len)), mode=self.mode)[:, :, shift_len:]
        return new_spec


#############################################################################
# Joint transform
class ComposeMapTransform:
    """
    Compose a list of data augmentation on numpy array. These data augmentation methods change both features and labels.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        for transform in self.transforms:
            x, y_sed, y_doa = transform(x, y_sed, y_doa)
        return x, y_sed, y_doa


class MapDataAugmentBase:
    """
    Base class for joint feature and label augmentation.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        if self.always_apply:
            return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
        else:
            if np.random.rand() < self.p:
                return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
            else:
                return x, y_sed, y_doa

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x: < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_sed: <np.ndarray (n_time_steps, n_classes)>
        :param y_doa: <np.ndarray (n_time_steps, 3*nclasses)>
        n_channels = 7 for salsa, melspeciv, linspeciv; 10 for melspecgcc, linspecgcc
        """
        raise NotImplementedError


class TfmapRandomSwapChannelFoa(MapDataAugmentBase):
    """
    This data augmentation random swap xyz channel of tfmap of FOA format. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, accdoa
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa
        This data augmentation change x_sed and y_doa
        x feature: x[-3]: Y, x[-2]: Z, x[-1]: X
            W Y Z X Y Z X: 7 channels
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 7, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(4,))
        # change input feature
        if m[0] == 1:  # random swap x, y
            x_new[1] = x[3]
            x_new[3] = x[1]
            x_new[-3] = x[-1]
            x_new[-1] = x[-3]
        if m[1] == 1:  # negate x
            x_new[-1] = -x_new[-1]
        if m[2] == 1:  # negate y
            x_new[-3] = -x_new[-3]
        if m[3] == 1:  # negate z
            x_new[-2] = -x_new[-2]
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1:
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, 0:self.n_classes]
            if m[2] == 1:
                y_doa_new[:, self.n_classes: 2*self.n_classes] = - y_doa_new[:, self.n_classes: 2*self.n_classes]
            if m[3] == 1:
                y_doa_new[:, 2*self.n_classes:] = - y_doa_new[:, 2*self.n_classes:]
        else:
            raise NotImplementedError('this output format not yet implemented')

        return x_new, y_sed, y_doa_new


class TfmapRandomSwapChannelMic(MapDataAugmentBase):
    """
    This data augmentation random swap channels of tfmap of MIC format.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, accdoa
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa, reg_accdoa
        This data augmentation change x and y_doa
        x: x[0]: M1, x[1] = M2, x[2]: M3, x[3]: M4
            M1 M2 M3 M4 p12 p13 p14: 7 channels
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 7, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(3,))
        # change inpute feature
        if m[0] == 1:  # swap M2 and M3 -> swap x and y
            x_new[1] = x[2]
            x_new[2] = x[1]
            x_new[-3] = x[-2]
            x_new[-2] = x[-3]
        if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
            x_cur = x_new.copy()
            x_new[0] = x_cur[3]
            x_new[3] = x_cur[0]
            x_new[-1] = - x_cur[-1]
            x_new[-2] = x_cur[-2] - x_cur[-1]
            x_new[-3] = x_cur[-3] - x_cur[-1]
        if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
            x_cur = x_new.copy()
            x_new[0] = x_cur[1]
            x_new[1] = x_cur[0]
            x_new[2] = x_cur[3]
            x_new[3] = x_cur[2]
            x_new[-3] = - x_cur[-3]
            x_new[-2] = x_cur[-1] - x_cur[-3]
            x_new[-1] = x_cur[-2] - x_cur[-3]
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1:  # swap M2 and M3 -> swap x and y
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
                temp = - y_doa_new[:, 0:self.n_classes].copy()
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, self.n_classes:2 * self.n_classes] = temp
            if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
                y_doa_new[:, self.n_classes:2 * self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, 2 * self.n_classes:] = - y_doa_new[:, 2 * self.n_classes:]
        else:
            raise NotImplementedError('this doa format not yet implemented')

        return x_new, y_sed, y_doa_new


class GccRandomSwapChannelMic(MapDataAugmentBase):
    """
    This data augmentation random swap channels of melspecgcc or linspecgcc of MIC format.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_classes: int = 12):
        super().__init__(always_apply, p)
        self.n_classes = n_classes

    def reflect_azi(self, azi, n_azis: int = 72):
        """reflect azi for eventwise clapolar format: azi -> -azi
        azi: (n_timesteps, n_azis, n_max_event)
        n_azis: even number"""
        azi = np.concatenate((np.flip(azi[:, n_azis//2 + 1:], axis=1),
                              np.flip(azi[:, :n_azis//2 + 1], axis=1)), axis=1)
        return azi

    def shift_azi(self, azi, azi_shift_deg, n_azis: int = 72):
        n_shifts = n_azis * azi_shift_deg // 360
        azi = np.concatenate((azi[:, -n_shifts:], azi[:, :-n_shifts]), axis=1)
        return azi

    def reflect_ele(self, ele, n_eles: int = 19):
        """reflect ele for eventwise clapolar format: ele -> -ele
        ele: (n_timesteps, n_eles, n_max_event)
        n_eles: odd number"""
        ele = np.concatenate((np.flip(ele[:, n_eles//2 + 1:], axis=1),
                              np.flip(ele[:, :n_eles//2 + 1], axis=1)), axis=1)
        return ele

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_sed: <np.ndarray (n_time_steps, n_classes)> reg_xyz, reg_polar, accdoa, reg_accdoa, cla_polar
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa, reg_accdoa
        This data augmentation change x and y_doa
        x: x[0]: M1, x[1] = M2, x[2]: M3, x[3]: M4
            M1 M2 M3 M4 xc12 xc13 xc14 xc23 xc24 xc34: 10 channels
        M1: n_timesteps x n_mels
        xc12: n_timesteps x n_lags (n_mels = n_lags)
        """
        n_input_channels = x.shape[0]
        assert n_input_channels == 10, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()
        # random method
        m = np.random.randint(2, size=(3,))
        if m[0] == 1:  # swap M2 and M3 -> swap x and y
            x_new[1] = x[2]
            x_new[2] = x[1]
            x_new[4] = x[5]
            x_new[5] = x[4]
            x_new[7] = np.flip(x[7], axis=-1)
            x_new[-1] = x[-2]
            x_new[-2] = x[-1]
        elif m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
            x_cur = x_new.copy()
            x_new[0] = x_cur[3]
            x_new[3] = x_cur[0]
            x_new[4] = np.flip(x_cur[8], axis=-1)
            x_new[5] = np.flip(x_cur[9], axis=-1)
            x_new[6] = np.flip(x_cur[6], axis=-1)
            x_new[8] = np.flip(x_cur[4], axis=-1)
            x_new[9] = np.flip(x_cur[5], axis=-1)
        elif m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
            x_cur = x_new.copy()
            x_new[0] = x_cur[1]
            x_new[1] = x_cur[0]
            x_new[2] = x_cur[3]
            x_new[3] = x_cur[2]
            x_new[4] = np.flip(x_cur[4], axis=-1)
            x_new[5] = x_cur[8]
            x_new[6] = x_cur[7]
            x_new[7] = x_cur[6]
            x_new[8] = x_cur[5]
            x_new[9] = np.flip(x_cur[9], axis=-1)
        # change y_doa
        if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
            if m[0] == 1: # swap M2 and M3 -> swap x and y
                y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes]
                y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes]
            if m[1] == 1:  # swap M1 and M4 -> swap x and y, negate x and y
                temp = - y_doa_new[:, 0:self.n_classes].copy()
                y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, self.n_classes:2 * self.n_classes] = temp
            if m[2] == 1:  # swap M1 and M2, M3 and M4 -> negate y and z
                y_doa_new[:, self.n_classes:2 * self.n_classes] = - y_doa_new[:, self.n_classes:2 * self.n_classes]
                y_doa_new[:, 2 * self.n_classes:] = - y_doa_new[:, 2 * self.n_classes:]
        else:
            raise NotImplementedError('this doa format not yet implemented')

        return x_new, y_sed, y_doa_new
