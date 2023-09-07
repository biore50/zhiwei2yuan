import random
import pickle
import mmcv
import numpy as np

random.seed(0)

data = '/home/lwy/data/liwuyan/Projectdir/normalized20/test_data.npy'
label = '/home/lwy/data/liwuyan/Projectdir/normalized20/test_label.pkl'

# label = np.load(label)
with open(label, 'rb') as f:
    sample_name, label = pickle.load(f)


data = np.load(data)

# output_train_pkl = '/home/lwy/data/liwuyan/Projectdir/normalized20/train.pkl'
output_val_pkl = '/home/lwy/data/liwuyan/Projectdir/normalized20/val.pkl'

n_samples = len(label)
results = []
prog_bar = mmcv.ProgressBar(n_samples)

for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 250
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (640, 480)
    anno['original_shape'] = (640, 480)
    anno['label'] = int(label[i])
    results.append(anno)
    prog_bar.update()

random.shuffle(results)

# total = 2922 split into train(2500) val(422)
# train_list = results
val_list = results
# print(f'len(train)={len(train_list)}')
print(f'len(val)={len(val_list)}')

# mmcv.dump(train_list, output_train_pkl)
mmcv.dump(val_list, output_val_pkl)
print('Finish!')
