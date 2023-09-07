import os
from sklearn.model_selection import train_test_split
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt




class VideoDataset(Dataset):

    def __init__(self, dataset='gcf', split='train', clip_len=16, preprocess=True):
        self.root_dir = './RGB'
        self.output_dir = './gcf/'
        self.label_dir = './label.txt'
        self.ske_dir = './ske/'
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.with_kp = True
        self.with_limb = False

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 60
        self.resize_width = 80
        self.crop_size = 56

        # if not self.check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You need to download it from official website.')
        #
        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf":
            if not os.path.exists(self.label_dir):
                with open(self.label_dir, 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer, img = self.load_frames(self.fnames[index])
        if self.split == 'train':
            # Perform data augmentation
            buffer = self.train_crop(buffer, self.clip_len, self.crop_size)
            img = self.train_clip(img, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.train_crop(buffer, self.clip_len, self.crop_size)
            img = self.train_clip(img, self.clip_len, self.crop_size)
            buffer = self.randomflip(buffer)
            img = self.randomflip(img)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        # img = self.normalize(img)
        img = self.to_tensor(img)
        return torch.from_numpy(buffer), torch.from_numpy(img), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 64 or np.shape(image)[1] != 64:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            # os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train, test = train_test_split(video_files, test_size=0.2, random_state=42)
            # train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            # val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            # if not os.path.exists(val_dir):
            #     os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            # for video in val:
            #     self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))



        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # file_ske = self.ske_dir + file_dir.split('/')[-3] + '/' + file_dir.split('/')[-2] + file_dir.split('/')[-1] + '.json'
        file_ske = os.path.join(self.ske_dir, file_dir.split('/')[-3],file_dir.split('/')[-2], file_dir.split('/')[-1] + '.json')
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        img = self.pose_loding(file_ske)

        return buffer, img

    def pose_loding(self, file_dir):
        sample_path = file_dir
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

            # fill data_numpy
        data = video_info['data']
        T = len(data)
        C = 3
        V = 28
        M = 1
        img_h=72
        img_w=72
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

        data_numpy = self.nozero(data_numpy, T)  # (t,v,3)

        all_kps = ((data_numpy[:, :, :2] + 1) * 50)[None]  # (1,T,V,2)
        all_kpscores = data_numpy[:, :, 2][None]  # (1,T,V)
        img_shape = (img_h, img_w) # (480, 640)

        img = self.gen_an_aug(all_kps, all_kpscores, img_shape)

        return img

    def nozero(self, x, T):
        score = x[..., -1]
        score_sum = np.sum(score, axis=-1)
        j = T - 1
        while score_sum[j] < 1e-2:
            j -= 1
        return x[:j + 1]

    def gen_an_aug(self, all_kps, all_kpscores, img_shape):
        use_score = True

        kp_shape = all_kps.shape
        all_kpscores = all_kpscores

        img_h, img_w = img_shape
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = 0.3
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)

            imgs.append(hmap)
        imgs = np.array(imgs)

        return imgs

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):

        skeletons = ((0, 1), (1, 2), (2, 3), (3, 17), (17, 18), (18, 19),
                     (17, 20), (20, 21), (17, 22), (22, 23), (17, 24), (24, 25),
                     (17, 26), (26, 27), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                     (6, 9), (9, 10), (6, 11), (11, 12), (6, 13), (13, 14),
                     (6, 15), (15, 16))

        left_kp = (2, 3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
        right_kp = (4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])


                heatmaps.append(heatmap)
        i=0
        if self.with_limb:
            for limb in skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)

                heatmaps.append(heatmap)

        return np.stack(heatmaps, axis=-1)



    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
        # max_values = np.squeeze(max_values) # (1,28) > (28,) change
        centers = np.squeeze(centers)  # (1,28,2) > (28,2) change
        mu_x, mu_y = centers[0], centers[1]
        # if max_values < self.eps:
        #     continue

        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)

        # if the keypoint not in the heatmap coordinate system
        # if not (len(x) and len(y)):
        #     continue
        y = y[:, None]

        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_values
        heatmap[st_y:ed_y,
        st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        eps = 1e-4
        heatmap = np.zeros([img_h, img_w], dtype=np.float32)
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

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                    end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = (
                    a_dominate * d2_start + b_dominate * d2_end +
                    seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma ** 2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap



    def train_crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        num_frames = buffer.shape[0]
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            time_index = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            bascic = np.arange(clip_len)
            time_index = np.random.choice(clip_len +1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[time_index] = 1
            offset = np.cumsum(offset)
            time_index = bascic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            time_index = bst + offset

        time_index = np.mod(time_index, num_frames)
         # start_index = 0
        time_index = time_index.astype(np.int32)

        if time_index.ndim !=1:
            time_index = np.squeeze(time_index)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :].astype(np.float32)

        return buffer

    def test_crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        num_clips = 1
        num_frames = buffer.shape[0]
        if num_frames < clip_len:
            if num_frames < num_clips:
                start = list(range(self.num_clips))
            else:
                start = [i * num_frames // num_clips for i in range(num_clips)]

            time_index = np.concatenate([np.arange(i, i + clip_len) for i in start])
        elif clip_len <= num_frames < 2 * clip_len:
            all_inds =[]
            for i in range(num_clips):
                bascic = np.arange(clip_len)
                time_index = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[time_index] = 1
                offset = np.cumsum(offset)
                time_index = bascic + offset[:-1]
                all_inds.append(time_index)
            time_index = np.concatenate(all_inds)
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            time_index = np.concatenate(all_inds)

        time_index = np.mod(time_index, num_frames)
         # start_index = 0
        time_index = time_index.astype(np.int32)

        if time_index.ndim !=1:
            time_index = np.squeeze(time_index)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :].astype(np.float32)

        return buffer

    def train_clip(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        num_frames = buffer.shape[0]
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            time_index = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            bascic = np.arange(clip_len)
            time_index = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[time_index] = 1
            offset = np.cumsum(offset)
            time_index = bascic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            time_index = bst + offset

        time_index = np.mod(time_index, num_frames)
        # start_index = 0
        time_index = time_index.astype(np.int32)

        if time_index.ndim != 1:
            time_index = np.squeeze(time_index)


        buffer = buffer[time_index, :, :, :].astype(np.float32)


        return buffer












if __name__ == "__main__":
    from torch.utils.data  import DataLoader
    a =1
    train_data = VideoDataset(dataset='ucf', split= 'train', clip_len=16, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=4)

    for i, inputs1,inputs2,inputs3 in enumerate(train_loader):
        # inputs1 = sample[0]
        # inputs2 = sample[1]
        labels = inputs3
        print('RGB=', inputs1.size())
        print('skele=', inputs2.size())
        print(labels)

        if i == 1 :
            break

