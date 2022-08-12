import torch
import torch.utils.data as data
import os
from scipy.spatial import distance
import numpy as np
from math import sqrt

# import localTransformations as transforms
import sys


def evaluate_matches(gt_keypoint_src, gt_keypoint_dst, matches, distance_matching_threshold):
    nb_false_matching, nb_true_matches, nb_rejected_matches = 0

    for j in range(0, len(gt_keypoint_src)):
        xp = int(gt_keypoint_dst[j][0])
        yp = int(gt_keypoint_dst[j][1])
        xp2 = int(gt_keypoint_dst[matches[j]][0])
        yp2 = int(gt_keypoint_dst[matches[j]][1])
        dist = sqrt((yp - yp2) ** 2 + (xp - xp2) ** 2)
        if matches[j] == -1:
            nb_rejected_matches += 1
        elif dist <= distance_matching_threshold:
            nb_true_matches += 1
        else:
            nb_false_matching += 1
    return nb_false_matching, nb_true_matches, nb_rejected_matches


def feature_extraction(patches, model, imageSize):
    list_desc = []
    model.eval()
    inputs1_tensors = torch.stack([torch.Tensor(i) for i in patches])

    for (idx, data) in enumerate(inputs1_tensors):

        inputs = data

        input_var = torch.autograd.Variable(inputs)

        input_var = input_var.view(1, imageSize, imageSize)

        if not (list(input_var.size())[0] == 1):
            continue

        inputs_var_batch = input_var.view(1, 1, imageSize, imageSize)
        # computed output
        with torch.no_grad():
            desc = model(inputs_var_batch).cuda()
        desc = torch.Tensor.cpu(desc).detach().numpy()
        list_desc.append(desc)

    return list_desc


def feature_match(feature_vectors_1, feature_vectors_2, matching_threshold):
    matched_idx = []
    distance_list = []
    for i in range(0, len(feature_vectors_1)):
        distance_sim = []
        for j in range(0, len(feature_vectors_2)):
            sim = distance.euclidean(feature_vectors_1[i], feature_vectors_2[j])
            distance_sim.append(sim)
        candidate_index = np.argsort(distance_sim)[:2]
        distance_list.extend([distance_sim[candidate_index[0]]])
        if distance_sim[candidate_index[0]] > matching_threshold:
            # set -1 if no match found for the current vector
            matched_idx.append(-1)
        else:
            matched_idx.extend([candidate_index[0]])

    return matched_idx, distance_list
