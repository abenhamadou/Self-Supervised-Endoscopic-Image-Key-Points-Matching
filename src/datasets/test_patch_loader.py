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

        self.all_frames_per_sequence = []
        self.all_keypoints = []
        self.data = []
        self.load_data_path()

    def __len__(self):
        return len(self.data)

    def load_data_path(self):
        # get list of  sequences
        sequence_list = glob.glob(self.root_path + "/*")

        # got through sequences
        for sequence in sequence_list:
            #  first get all gt matches for the current sequence
            curr_matches = sorted(glob.glob(sequence+"/matches/*"))
            for match in curr_matches:
                # get src and dst frame filenames
                _, src_filename, dst_filename = osp.splitext(osp.split(match)[1])[0].split("_")
                src_abs_path = osp.join(sequence, "frames", src_filename)
                dst_abs_path = osp.join(sequence, "frames", dst_filename)

                # if all related data exist
                if osp.exists(src_abs_path) and osp.exists(dst_abs_path):
                    self.data.append({"match": match, "src_frame": src_abs_path, "dst_frame": dst_abs_path})

    def enhance(self, img):
        crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 40]
        gray2 = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        # image_enhanced = cv.equalizeHist(gray2)
        return image_enhanced

    def __getitem__(self, idx):
        frame_src = cv.imread(self.data[idx]["src_frame"], 3)
        Next_frame = cv.imread(self.data[idx]["dst_frame"], 3)

        frame_src = self.enhance(frame_src)
        Next_frame = self.enhance(Next_frame)

        # keypoints filenames
        matches = self.data[idx]["match"]

        list_matches = loadtxt(matches, dtype='int')

        list_keypoints_src = []
        list_keypoints_next_frame = []

        for i in range(0, len(list_matches)):
            list_keypoints_src.append((list_matches[i][0], list_matches[i][1]))
            list_keypoints_next_frame.append((list_matches[i][2], list_matches[i][3]))
        h, w = frame_src.shape

        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        i = 0
        gt_key_src = []
        gt_key_next_frame = []
        patches_src = []
        patches_next_frame = []
        i = 0
        # ---------------------------------------------------Generate_data-----------------------------------------------------------

        for b in range(0, len(list_keypoints_src)):

            xa = int(list_keypoints_src[b][0])
            ya = int(list_keypoints_src[b][1])
            xp = int(list_keypoints_next_frame[b][0])
            yp = int(list_keypoints_next_frame[b][1])

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
                gt_key_src.append(list_keypoints_src[i])
                gt_key_next_frame.append(list_keypoints_next_frame[i])
                i += 1
        return {
            "image_src_name": frame_src,
            "image_dst_name": Next_frame,
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            "keypoint_dst": gt_key_next_frame,
            "keypoint_src": gt_key_src
        }