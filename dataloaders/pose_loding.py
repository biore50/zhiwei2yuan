import torch
import json
import os
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import torch

def pose_loding(file_dir):
    sample_path = file_dir
    with open(sample_path, 'r') as f:
        video_info = json.load(f)

        # fill data_numpy
    data = video_info['data']
    T = len(data)
    C = 3
    V = 28
    M = 1
    data_numpy = np.zeros((C, T, V, M))
    for frame_info in data:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= M:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    # sort by score
    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
    # (C,T,V,M)
    data_numpy = data_numpy[:, :, :, 0:M].transpose((1, 2, 0, 3)).astype(np.float16)
    data_numpy = np.squeeze(data_numpy)  # (T,V,C)

    data_numpy = nozero(data_numpy, T)   # (t,v,3)

    all_kps = ((data_numpy[:, :, :2] + 1) * 50)[None] # (1,T,V,2)
    all_kpscores = data_numpy[:, :, 2][None]  # (1,T,V)
    img_shape = (72, 72)

    img = gen_an_aug(all_kps,all_kpscores,img_shape)

    return img


def nozero(x,T):
    score = x[..., -1]
    score_sum = np.sum(score, axis=-1)
    j = T-1
    while score_sum[j] < 1e-2:
        j -= 1
    return x[:j + 1]

def gen_an_aug(all_kps,all_kpscores,img_shape):
        """Generate pseudo heatmaps for all frames.
        Args:
            results (dict): The dictionary that contains all info of a sample.
        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        # all_kps = results[0]['keypoint']
        use_score = True

        skeletons = ((0, 1), (1, 2), (2, 3), (3, 17), (17, 18), (18, 19),
                     (17, 20), (20, 21), (17, 22), (22, 23), (17, 24), (24, 25),
                     (17, 26), (26, 27), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                     (6, 9), (9, 10), (6, 11), (11, 12), (6, 13), (13, 14),
                     (6, 15), (15, 16))
        kp_shape = all_kps.shape


        # if 'keypoint_score' in results[1]:
        #     all_kpscores = results[1]['keypoint_score']
        # else:
        #     all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)


        img_h, img_w = img_shape
        num_frame = kp_shape[1]
        imgs = []
        num_c = 0
        for i in range(num_frame):
            sigma = 0.3
            # M, V, C
            kps = all_kps[:, i]
            kpscores = all_kpscores[:,i]
            max_values = np.ones(kpscores.shape, dtype=np.float32)
            # M, C
            if use_score:
                max_values = kpscores

            heatmap = generate_heatmap(img_h, img_w, kps, sigma, max_values)

            map = np.uint8(255 * heatmap)
            plt.subplot(1, 2, 1)
            # plt.rcParams['savefig.dpi']= 300
            plt.imshow(map)
            # plt.show()
            save = '/home/lwy/projects/PoseC3D/figs/afraid/keymap{}.jpg'.format(i)
            plt.savefig(save)
            imgs.append(heatmap)

        return imgs


def generate_a_heatmap( heatmap, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        eps = 1e-4
        sigma = 0.6
        # heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        img_h, img_w = heatmap.shape
        # max_values = np.squeeze(max_values)  # (1,28) > (28,) change

        mu_x, mu_y = centers[0], centers[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)

        for y in range(st_y, ed_y,):
            for x in range(st_x, ed_x):
                patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
                patch = patch * max_values
                heatmap[y, x] = np.maximum(heatmap[y, x], patch)
                heatmap[y, x] = np.minimum(heatmap[y, x], 1.0)
        # x = np.arange(st_x, ed_x, 1, np.float32)
        # y = np.arange(st_y, ed_y, 1, np.float32)
        #
        # # if the keypoint not in the heatmap coordinate system
        # # if not (len(x) and len(y)):
        #
        # y = y[:, None]
        #
        # patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        # patch = patch * max_values
        # heatmap[st_y:ed_y, st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x], patch)
        # for center, max_value in zip(centers, max_values):
        #     if max_value < eps:
        #         continue
        # plt.imshow(heatmap)
        return heatmap



def generate_a_limb_heatmap(heatmap, starts, ends, sigma, start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        eps = 1e-4
        sigma = 0.3
        img_h, img_w = heatmap.shape

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

            # distance to end keypoints
            d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

            # if d2_ab < 1:
            #     generate_a_heatmap(heatmap, start[None], sigma, start_value[None])
            #     continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2. / sigma ** 2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(heatmap[min_y:max_y, min_x:max_x], patch)
        # plt.imshow(heatmap)


def generate_heatmap(img_h, img_w,  kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).
        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        with_kp = True
        with_limb = False
        skeletons = ((0, 1), (1, 2), (2, 3), (3, 17), (17, 18), (18, 19),
                     (17, 20), (20, 21), (17, 22), (22, 23), (17, 24), (24, 25),
                     (17, 26), (26, 27), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                     (6, 9), (9, 10), (6, 11), (11, 12), (6, 13), (13, 14),
                     (6, 15), (15, 16))

        left_kp = (2, 3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
        right_kp = (4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

        heatmaps = []
        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        if with_kp:
            num_kp = kps.shape[1]
            # heatmap = np.zeros([num_kp,img_h, img_w], dtype=np.float32)
            for i in range(num_kp):
                point = kps[:,i]
                point = np.squeeze(point)
                if point[0] < 0 or point[1] < 0:
                    continue

                max_value = max_values[:, i]

                generate_a_heatmap(heatmap, point, sigma, max_value)


                # heatmaps.append(heatmap)

        if with_limb:
            for i, limb in enumerate(skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                generate_a_limb_heatmap(heatmap, starts, ends, sigma, start_values, end_values)

                # map = np.uint8(255 * heatmap)
                # plt.subplot(1, 2, 1)
                # plt.imshow(map)
                # plt.show()
                # save = '/home/lwy/data/liwuyan/Projectdir/PoseC3D/figs/limbmap{}.jpg'.format(i)
                # plt.savefig(save)
                # i = i+1
                #
                # heatmaps.append(heatmap)
        # plt.imshow(heatmap)

        return heatmap # np.stack(heatmaps, axis=-1)





if __name__ == "__main__":
    from torch.utils.data  import DataLoader

    file_dir = '/home/lwy/projects/Dataset/BASL20/ske20/test/afraid/afraid-002.json'
    data = pose_loding(file_dir= file_dir)
    print(data.shape)





