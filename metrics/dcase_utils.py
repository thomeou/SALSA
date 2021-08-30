"""
utilities functions for dcase submission format
"""

import numpy as np


def load_output_format_file(_output_format_file, version='2021'):
    """
    copy from cls_feature_class: remove class params
    Loads DCASE output format csv file and returns it in dictionary format
    can load both polar and xyz format

    params:
        _output_format_file: DCASE output format CSV
        submission output format: [frame_index, sound_class_idx, azimuth(degree), elevation(degree)]
        ground truth format: [frame_index, sound_class_idx, track_num, azimuth(degree), elevation(degree)]
        baseline format: [frame_index, sound_class_idx, track_num, x, y, z]
        version: choice: '2020', '2021', version '2021' includes track_num in the dictionary but this varialbe is ignored.
    
    return:
        _output_dict: dictionary
    """
    
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    if version == '2021':
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4: # output format of submission files
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[2]), float(_words[3]), 0])
            elif len(_words) == 5: #read polar coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), int(_words[2])])
            elif len(_words) == 6: # read Cartesian coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5]),
                                                 int(_words[2])])
    elif version == '2020':
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4: # output format of submission files
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[2]), float(_words[3])])
            elif len(_words) == 5: #read polar coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4])])
            elif len(_words) == 6: # read Cartesian coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
    else:
        raise ValueError('version {} is not implemented'.format(version))
    _fid.close()
    
    return _output_dict


def convert_output_format_polar_to_cartesian(in_dict, version='2021'):
    ''' 
    copy from cls_feature_class, remove class params
    convert polar format in degree to cartesian format'''
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[2]*np.pi/180.
                azi_rad = tmp_val[1]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                if version == '2021':
                    out_dict[frame_cnt].append([tmp_val[0], x, y, z, tmp_val[-1]])
                elif version == '2020':
                    out_dict[frame_cnt].append([tmp_val[0], x, y, z])
                else:
                    raise ValueError('version {} is not implemented'.format(version))

    return out_dict


def convert_output_format_cartesian_to_polar(in_dict, version='2021'):
    ''' convert cartesian format to polar format in degree'''
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x = tmp_val[1]
                y = tmp_val[2]
                z = tmp_val[3]
                
                azi_deg = np.arctan2(y,x) * 180.0/np.pi
                ele_deg = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180.0/np.pi

                if version == '2021':
                    out_dict[frame_cnt].append([tmp_val[0], azi_deg, ele_deg, tmp_val[-1]])
                elif version == '2020':
                    out_dict[frame_cnt].append([tmp_val[0], azi_deg, ele_deg])
                else:
                    raise ValueError('version {} is not implemented'.format(version))


    return out_dict


def output_format_to_regression_format(output_dict, doa_output_format='polar', n_classes=14, n_max_frames=600,
                                       version='2021'):
    ''' convert output format in dictionary to regression output format, this will overwite some segments
    where events have the same classes.
    params:
        output_dict: key: frame index
                     values: [sound_class_idx, azimuth (degrees), elevation (degrees)] or
                             [sound_class_idx, x, y, z]
                     n_max_frames < label_frames_per_1s * file_len_s
        doa_input_format (infer): 'polar' (degree) | 'xyz'
        doa_output_format: (str) 'polar' (degree) | 'xyz'
        polar -> polar,  xyz
        xyz -> polar, xyz
    returns:
        [sed_output, doa_output]
        sed_output: (numpy.array) [n_max_frames, n_classes]
        doa_output: (numpy.array) [n_max_frames, 2 * n_classes] if doa_format is polar ('degree')
                                  [n_max_frames, 3 * n_classes] if doa_format is 'xyz'
    '''
    # n_max_frames = int(label_frames_per_1s * file_len_s)
    sed_output = np.zeros((n_max_frames, n_classes))
    if doa_output_format == 'xyz':
        doa_output = np.zeros((n_max_frames, n_classes*3))
    else:
        doa_output = np.zeros((n_max_frames, n_classes*2))
        
    count = 0   
    for frame_idx, values in output_dict.items():
        if frame_idx < n_max_frames:
            for value in values:  
                if count == 0:
                    if version == '2020':
                        if len(value) == 3:
                            doa_input_format = 'polar'
                        elif len(value) == 4:
                            doa_input_format = 'xyz'
                    elif version == '2021':
                        if len(value) == 3 or len(value) == 4:
                            doa_input_format = 'polar'
                        elif len(value) == 5:
                            doa_input_format == 'xyz'
                    else:
                        raise ValueError('Version {} is unknown'.format(version))
                    count += 1
                sound_class_idx = int(value[0])
                sed_output[frame_idx, sound_class_idx] = 1
                if doa_input_format == 'polar' and doa_output_format == 'polar':
                    doa_output[frame_idx, sound_class_idx] = value[1]
                    doa_output[frame_idx, n_classes + sound_class_idx] = value[2]
                elif doa_input_format == 'polar' and doa_output_format == 'xyz':                    
                    azi_rad = value[1]*np.pi/180
                    ele_rad = value[2]*np.pi/180.
                    x = np.cos(azi_rad) * np.cos(ele_rad)
                    y = np.sin(azi_rad) * np.cos(ele_rad)
                    z = np.sin(ele_rad)
                    doa_output[frame_idx, sound_class_idx] = x
                    doa_output[frame_idx, n_classes + sound_class_idx] = y
                    doa_output[frame_idx, 2*n_classes + sound_class_idx] = z             
                elif doa_input_format == 'xyz' and doa_output_format == 'polar':
                    x = value[1]
                    y = value[2]
                    z = value[3]
                    azi_rad = np.arctan2(y, x)
                    ele_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
                    doa_output[frame_idx, sound_class_idx] = azi_rad * 180.0/np.pi
                    doa_output[frame_idx, n_classes + sound_class_idx] = ele_rad * 180.0/np.pi                
                else: #elif doa_input_format == 'xyz' and doa_output_format == 'xyz':
                    doa_output[frame_idx, sound_class_idx] = value[1]
                    doa_output[frame_idx, n_classes + sound_class_idx] = value[2]
                    doa_output[frame_idx, 2*n_classes + sound_class_idx] = value[3]   
    return [sed_output, doa_output]


def segment_labels(_pred_dict, _max_frames=600, _nb_label_frames_1s=10):
    '''
    Same for both 2021 and 2020 evaluation metrics
    copy form cls_feature_class: remove class params
    Collects class-wise sound event location information in segments of length 1s from reference dataset
    :param 
        _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        _max_frames: Total number of frames in the recording
        _nb_label_frames_1s: label frame rate or number of frame per second for label 
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    nb_blocks = int(np.ceil(_max_frames/float(_nb_label_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, _nb_label_frames_1s):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // _nb_label_frames_1s
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+_nb_label_frames_1s):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}  # key of loc_dict: dict[class_idx][block_frame] = [azi, ele, track_idx]

                block_frame = audio_frame - frame_cnt  # block_frame range: [0, 10)
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]  # keys: list of block_frames
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]] # values: list of [azi, ele, track_idx]

            output_dict[block_cnt][class_cnt].append([keys, values])  # output_dict[block_idx][class_idx] = [[keys, values]]

    return output_dict


def regression_label_format_to_output_format(_sed_labels, _doa_labels, _nb_classes=14):
        """
        copy form cls_feature_class remove class parma
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict


def output_format_dict_to_classification_labels(output_dict, azimuths, elevations,
                                                n_classes=14, n_max_frames_per_file=600,
                                                joint=True):
    '''
    if joint is True, return [n_max_frames_per_file, n_classes, n_azimuths * n_elevations]
    else: return             [n_max_frames_per_file, n_classes, n_azimuths, n_elevations]
    output dict:
            key: frame_idx
            values: [sound_class_idx, azimuth (degrees), elevation (degrees)]
    returns:
        classification format:[n_max_frames_per_file, n_classes, n_azimuths * n_elevations]'''
    
    n_azis = len(azimuths)
    n_eles = len(elevations)
    azi_reln = int(abs(azimuths[1] - azimuths[0]))
    ele_reln = int(abs(elevations[1] - elevations[0]))

    if joint:
        labels = np.zeros((n_max_frames_per_file, n_classes, n_azis * n_eles))
    else:
        labels = np.zeros((n_max_frames_per_file, n_classes, n_azis, n_eles))

    for frame_idx in output_dict.keys():
        if frame_idx <= n_max_frames_per_file:
            for value in output_dict[frame_idx]:
                # Making sure the doa's are within the limits
                azi = np.clip(value[1], azimuths[0], azimuths[-1])
                ele = np.clip(value[2], elevations[0], elevations[-1])
                if joint:
                    doa_idx = int(azi - azimuths[0])//azi_reln * n_eles + int(ele-elevations[0])//ele_reln
                    # create label
                    labels[frame_idx, value[0], int(doa_idx)] = 1
                else:
                    azi_idx = int((azi - azimuths[0])//azi_reln)
                    ele_idx = int((ele - elevations[0])//ele_reln)
                    labels[frame_idx, value[0], azi_idx, ele_idx] = 1

    return labels
