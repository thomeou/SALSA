"""
Copied from https://github.com/sharathadavanne/seld-dcase2020/blob/master/metrics/evaluation_metrics.py
"""
#
# Implements the localization and detection metrics proposed in the paper
#
# Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
#
# This script has MIT license
#

import numpy as np
from IPython import  embed
eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''

        self._TP = 0
        self._FP = 0
        self._TN = 0
        self._FN = 0

        self._S = 0
        self._D = 0
        self._I = 0

        self._Nref = 0
        self._Nsys = 0

        self._total_DE = 0
        self._DE_TP = 0

        self._spatial_T = doa_threshold
        self._nb_classes = nb_classes

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # Location-senstive detection performance
        ER = (self._S + self._D + self._I) / float(self._Nref + eps)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        if self._DE_TP:
            DE = self._total_DE / float(self._DE_TP + eps)
        else:
            # When the total number of prediction is zero
            DE = 180

        DE_prec = float(self._DE_TP) / float(self._Nsys + eps)
        DE_recall = float(self._DE_TP) / float(self._Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        aux_metrics = [self._S/float(self._Nref + eps),
                       self._D / float(self._Nref + eps),
                       self._I / float(self._Nref + eps),
                       prec,
                       recall,
                       DE_prec,
                       DE_recall,
                       self._Nsys/self._Nref]

        return ER, F, DE, DE_F
        # return ER, F, DE, DE_F, aux_metrics

    def update_seld_scores_xyz(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation [5] in the paper, using Cartesian distance

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            # print('\nblock_cnt', block_cnt, end='')
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # print('\tclass:', class_cnt, end='')
                # Counting the number of ref and sys outputs should include the number of tracks for each class in the segment
                if class_cnt in gt[block_cnt]:
                    self._Nref += 1
                if class_cnt in pred[block_cnt]:
                    self._Nsys += 1

                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False negative case

                    # NOTE: For multiple tracks per class, identify multiple tracks using hungarian algorithm and then
                    # calculate the spatial distance using the following code. In the current code, if there are multiple 
                    # tracks of the same class in a frame we are calculating the least cost between the groundtruth and predicted and using it.

                    total_spatial_dist = 0
                    total_framewise_matching_doa = 0
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            total_framewise_matching_doa += 1
                            pred_ind = pred_ind_list.index(gt_val)

                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])

                            if gt_arr.shape[0]==1 and pred_arr.shape[0]==1:
                                total_spatial_dist += distance_between_cartesian_coordinates(gt_arr[0][0], gt_arr[0][1], gt_arr[0][2], pred_arr[0][0], pred_arr[0][1], pred_arr[0][2])
                            else:
                                total_spatial_dist += least_distance_between_gt_pred(gt_arr, pred_arr)

                    if total_spatial_dist == 0 and total_framewise_matching_doa == 0:
                        loc_FN += 1
                        self._FN += 1
                    else:
                        avg_spatial_dist = (total_spatial_dist / total_framewise_matching_doa)

                        self._total_DE += avg_spatial_dist
                        self._DE_TP += 1

                        if avg_spatial_dist <= self._spatial_T:
                            self._TP += 1
                        else:
                            loc_FN += 1
                            self._FN += 1
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self._FN += 1
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self._FP += 1
                elif class_cnt not in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # True negative
                    self._TN += 1

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return

    def update_seld_scores(self, pred_deg, gt_deg):
        '''
        Implements the spatial error averaging according to equation [5] in the paper, using Polar distance
        Expects the angles in degrees

        :param pred_deg: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt_deg: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt_deg.keys())):
            # print('\nblock_cnt', block_cnt, end='')
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # print('\tclass:', class_cnt, end='')
                # Counting the number of ref and sys outputs should include the number of tracks for each class in the segment
                if class_cnt in gt_deg[block_cnt]:
                    self._Nref += 1
                if class_cnt in pred_deg[block_cnt]:
                    self._Nsys += 1

                if class_cnt in gt_deg[block_cnt] and class_cnt in pred_deg[block_cnt]:
                    # True positives or False negative case

                    # NOTE: For multiple tracks per class, identify multiple tracks using hungarian algorithm and then
                    # calculate the spatial distance using the following code. In the current code, if there are multiple 
                    # tracks of the same class in a frame we are calculating the least cost between the groundtruth and predicted and using it.
                    total_spatial_dist = 0
                    total_framewise_matching_doa = 0
                    gt_ind_list = gt_deg[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred_deg[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            total_framewise_matching_doa += 1
                            pred_ind = pred_ind_list.index(gt_val)

                            gt_arr = np.array(gt_deg[block_cnt][class_cnt][0][1][gt_ind]) * np.pi / 180
                            pred_arr = np.array(pred_deg[block_cnt][class_cnt][0][1][pred_ind]) * np.pi / 180
                            if gt_arr.shape[0]==1 and pred_arr.shape[0]==1:
                                total_spatial_dist += distance_between_spherical_coordinates_rad(gt_arr[0][0], gt_arr[0][1], pred_arr[0][0], pred_arr[0][1])
                            else:
                                total_spatial_dist += least_distance_between_gt_pred(gt_arr, pred_arr)

                    if total_spatial_dist == 0 and total_framewise_matching_doa == 0:
                        loc_FN += 1
                        self._FN += 1
                    else:
                        avg_spatial_dist = (total_spatial_dist / total_framewise_matching_doa)

                        self._total_DE += avg_spatial_dist
                        self._DE_TP += 1

                        if avg_spatial_dist <= self._spatial_T:
                            self._TP += 1
                        else:
                            loc_FN += 1
                            self._FN += 1
                elif class_cnt in gt_deg[block_cnt] and class_cnt not in pred_deg[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self._FN += 1
                elif class_cnt not in gt_deg[block_cnt] and class_cnt in pred_deg[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self._FP += 1
                elif class_cnt not in gt_deg[block_cnt] and class_cnt not in pred_deg[block_cnt]:
                    # True negative
                    self._TN += 1

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
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
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
        :return: cost -  distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    return cost


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
        doa_error[0]/180,
        1 - doa_error[1]]
        )
    return seld_metric
