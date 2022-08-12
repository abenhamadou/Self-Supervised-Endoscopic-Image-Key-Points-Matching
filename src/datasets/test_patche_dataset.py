import numpy as np
import torch
import os
import cv2 as cv
from torch.utils.data import Dataset
from scipy.spatial import distance
from numpy import loadtxt
import glob
import os.path as osp


class PatchDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, root_path, patch_size):
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.patch_size = patch_size
        self.sequence_list = glob.glob(root_path+"/*")
        self.all_frames_per_sequence = []
        self.all_keypoints = []

        for seq in self.sequence_list:
            self.all_frames_per_sequence.append(glob.glob(seq+"/frames/*"))

        # get list of key-points corresponding to the list of frames
        for frames in self.all_frames_per_sequence:
            osp.

        self.keypoints_GT += [Path_test + "/keypoints_GT/" + f for f in os.listdir(Path_test + "/keypoints_GT/")]

    def __len__(self):
        return len(self.image_name)

    def enhance(self, img):
        crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 40]
        gray2 = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        # image_enhanced = cv.equalizeHist(gray2)
        return image_enhanced

    def __getitem__(self, idx):

        image_name = sorted(self.image_name)

        frame_src = cv.imread(image_name[idx], 3)

        Next_frame = cv.imread(image_name[idx + 1], 3)
        frame_src = self.enhance(frame_src)
        Next_frame = self.enhance(Next_frame)


        # -------------------------------------------------------GT_keypoints------------------------------------------------------------------
        GT_keypoint_src = self.Path_test + "/keypoints_GT/GT_kaypoint1_%i.txt" % (idx + 1)
        GT_keypoint_Next_frame = self.Path_test + "/keypoints_GT/GT_kaypoint2_%i.txt" % (idx + 1)

        list_GT_keypoint_src = loadtxt(GT_keypoint_src, dtype="int")
        list_GT_keypoint_dst = loadtxt(GT_keypoint_Next_frame, dtype="int")
        key_GT1 = []
        key_GT2 = []
        for i in range(0, len(list_GT_keypoint_src)):
            distancesim = []
            for j in range(0, len(sift_key_src)):
                dist = distance.euclidean(list_GT_keypoint_src[i], sift_key_src[j])
                distancesim.append(dist)
            candidate_index = np.argsort(distancesim)[:1]

            if (distancesim[candidate_index[0]]) < 5:
                key_GT1.append(list_GT_keypoint_src[i])
                key_GT2.append(list_GT_keypoint_dst[i])

        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        i = 0
        GT_key_src = []
        GT_key_next_frame = []
        patches_src = []
        patches_next_frame = []
        i = 0
        for b in range(0, len(key_GT1)):
            xa = int(key_GT1[b][0])
            ya = int(key_GT1[b][1])
            xp = int(key_GT2[b][0])
            yp = int(key_GT2[b][1])
            # cv.circle(image_enhanced, (xa, ya), 3, (0, 0, 0), -10)
            if (
                ((ya - self.patch_size) > 0)
                & ((xa - self.patch_size) > 0)
                & ((ya + self.patch_size) < h)
                & ((xa + self.patch_size) < w)
                & ((yp - self.patch_size) > 0)
                & ((xp - self.patch_size) > 0)
                & ((yp + self.patch_size) < h)
                & ((xp + self.patch_size) < w)
            ):
                crop_patches_src = frame_src[
                    ya - self.patch_size : ya + self.patch_size, xa - self.patch_size : xa + self.patch_size
                ]
                crop_patches_next_frame = Next_frame[
                    yp - self.patch_size : yp + self.patch_size, xp - self.patch_size : xp + self.patch_size
                ]
                patches_next_frame.append(crop_patches_next_frame)
                patches_src.append(crop_patches_src)
                GT_key_src.append(key_GT1[i])
                GT_key_next_frame.append(key_GT2[i])
                i += 1
        return {
            "image_src_name": frame_src,
            "image_dst_name": Next_frame,
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            "keypoint_dst": GT_key_src,
            "keypoint_src": GT_key_next_frame,
        }
