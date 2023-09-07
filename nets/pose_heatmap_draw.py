import copy as cp
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GPoseT(nn.Module):
    """Generate pseudo heatmaps based on joint coordinates and confidence.
    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".
    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """
    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (1, 2), (2, 3), (3, 17), (17, 18), (18, 19),
                            (17, 20), (20, 21), (17, 22), (22, 23), (17, 24), (24, 25),
                            (17, 26), (26, 27), (1, 4), (4, 5), (5, 6),(6, 7), (7, 8),
                            (6, 9), (9, 10), (6, 11), (11, 12), (6, 13), (13, 14),
                            (6, 15), (15, 16)),
                 double=False,
                 left_kp=(2, 3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
                 right_kp=(4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)):
        super(GPoseT, self).__init__()

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons


    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
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

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        max_values = np.squeeze(max_values) # (1,28) > (28,) change
        centers = np.squeeze(centers)  # (1,28,2) > (28,2) change

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
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

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        starts = np.squeeze(starts)  # (1,28) > (28,) change
        ends = np.squeeze(ends)  # (1,28,2) > (28,2) change
        start_values = np.squeeze(start_values)  # (1,28) > (28,) change
        end_values = np.squeeze(end_values)  # (1,28,2) > (28,2) change

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
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
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            # if d2_ab < 1:
            #     full_map = self.generate_a_heatmap(img_h, img_w, [start],
            #                                        sigma, [start_value])
            #     heatmap = np.maximum(heatmap, full_map)
            #     continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
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

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):

                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                map = np.uint8(255 * heatmap)
                plt.subplot(1,2,1)
                plt.imshow(map)
                # plt.show()
                save = '/home/lwy/data/liwuyan/Projectdir/PoseC3D/figs/keymap{}.jpg'.format(i)
                plt.savefig(save)

                heatmaps.append(heatmap)
        i = 0
        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                # map = np.uint8(255 * heatmap)
                # plt.subplot(1, 2, 1)
                # plt.imshow(map)
                # # plt.show()
                # save = '/home/lwy/data/liwuyan/Projectdir/PoseC3D/figs/limbmap{}.jpg'.format(i)
                # plt.savefig(save)
                # i = i+1

                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)




if __name__ == '__main__':
    import pickle
    import numpy as np
    import torch
    from nets.pose_heatmap_draw import GPoseT



    data = '/home/lwy/data/liwuyan/Projectdir/normalized20/PoseC3d/skival.pkl'

    with open(data, 'rb') as f:
        data = pickle.load(f)


    sigma = 0.6
    kps = data[0]['keypoint']
    max_values = data[1]['keypoint_score']
    img = data[3]['img_shape']
    img_h = img[0]
    img_w = img[1]

    skeletons = ((0, 1), (1, 2), (2, 3), (3, 17), (17, 18), (18, 19),
                 (17, 20), (20, 21), (17, 22), (22, 23), (17, 24), (24, 25),
                 (17, 26), (26, 27), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                 (6, 9), (9, 10), (6, 11), (11, 12), (6, 13), (13, 14),
                 (6, 15), (15, 16))

    left_kp = (2, 3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
    right_kp = (4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    G = GPoseT(sigma=0.6, skeletons=skeletons, left_kp=left_kp, right_kp=right_kp)
    heatmap = []
    heatmap = G.generate_heatmap(img_h, img_w, kps, sigma, max_values)

    print(heatmap.shape)
















