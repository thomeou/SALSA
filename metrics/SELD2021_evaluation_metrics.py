# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
#
# [1] Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen.
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020).
#
# This script has MIT license
#

import numpy as np

eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment
from IPython import embed


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        self._TP = 0
        self._FP = 0
        self._FN = 0
        self._TPc = np.zeros((nb_classes,))
        self._FPc = np.zeros((nb_classes,))
        self._FNc = np.zeros((nb_classes,))

        self._S = 0
        self._D = 0
        self._I = 0
        self._Nref = 0

        self._spatial_T = doa_threshold

        # Variables for Class-sensitive localization performance
        self._total_DE = 0

        self._DE_TP = 0
        self._DE_FP = 0
        self._DE_FN = 0

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # Location-sensitive detection performance
        ER = (self._S + self._D + self._I) / float(self._Nref + eps)
        F = self._TP / (eps + self._TP + 0.5 * (self._FP + self._FN))

        # Class-sensitive localization performance
        LE = self._total_DE / float(
            self._DE_TP + eps) if self._DE_TP else 180  # When the total number of prediction is zero
        LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)
        # print('S {}, D {}, I {}, Nref {}, TP {}, FP {}, FN {}, DE_TP {}, DE_FN {}, totalDE {}'.format(self._S, self._D, self._I, self._Nref, self._TP, self._FP, self._FN, self._DE_TP, self._DE_FN, self._total_DE))

        # SED stats
        error_stats = [self._S, self._D, self._I, self._Nref]
        f1_stats = [self._TP, self._FP, self._FN]
        # DOA stats
        doa_stats = [self._DE_TP, self._DE_FN]

        return ER, F, LE, LR
        # return ER, F, LE, LR, error_stats, f1_stats, doa_stats  # For error analysis
        # return ER, F, LE, LR, error_stats, f1_stats, doa_stats, self._TPc, self._FPc, self._FNc  # class-wise output

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt can either both be Cartesian or Degrees

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[
                    block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[
                    block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_cnt, gt_ind in enumerate(gt_ind_list):
                        if gt_ind in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_cnt])
                            gt_ids = np.arange(
                                len(gt_arr[:, -1]))  # TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, :-1]

                            pred_ind = pred_ind_list.index(gt_ind)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, :-1]

                            if gt_doas.shape[-1] == 2:  # convert DOAs to radians, if the input is in degrees
                                gt_doas = gt_doas * np.pi / 180.
                                pred_doas = pred_doas * np.pi / 180.

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN += nb_pred_doas
                        self._FNc[class_cnt] +=nb_pred_doas
                        self._DE_FN += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            self._total_DE += avg_spatial_dist
                            self._DE_TP += 1

                            # Location-sensitive detection performance
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP += 1
                                self._TPc[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP += 1
                                self._FPc[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive
                            loc_FP += (nb_pred_doas - nb_gt_doas)
                            self._FP += (nb_pred_doas - nb_gt_doas)
                            self._FPc[class_cnt] += (nb_pred_doas - nb_gt_doas)
                            self._DE_FP += (nb_pred_doas - nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative
                            loc_FN += (nb_gt_doas - nb_pred_doas)
                            self._FN += (nb_gt_doas - nb_pred_doas)
                            self._FNc[class_cnt] += (nb_gt_doas - nb_pred_doas)
                            self._DE_FN += (nb_gt_doas - nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN += nb_gt_doas
                    self._FNc[class_cnt] += nb_gt_doas
                    self._DE_FN += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP += nb_pred_doas
                    self._FPc[class_cnt] += nb_pred_doas
                    self._DE_FP += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2 + 1e-10)
    N2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1 / N1, y1 / N1, z1 / N1, x2 / N2, y2 / N2, z2 / N2

    # Compute the distance
    dist = x1 * x2 + y1 * y2 + z1 * z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3:  # Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[
                ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[
                                         ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[
                ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2,
                                                                                                    ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind


def early_stopping_metric(sed_error, doa_error):
    """
    Compute early stopping metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: early stopping metric result
    """
    seld_metric = np.mean([
        sed_error[0],
        1 - sed_error[1],
        doa_error[0] / 180,
        1 - doa_error[1]]
    )
    return seld_metric



