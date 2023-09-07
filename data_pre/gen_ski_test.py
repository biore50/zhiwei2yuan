import mmcv
import numpy as np

output_pkl = '/home/lwy/data/liwuyan/Projectdir/normalized20/ski/test.pkl'
test_data = '/home/lwy/data/liwuyan/Projectdir/normalized20/test_data.npy'

data = np.load(test_data)
n_samples = len(data)
print(n_samples)

results = []
prog_bar = mmcv.ProgressBar(n_samples)

for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 250
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (640, 480)
    anno['original_shape'] = (640, 480)
    anno['label'] = 0
    results.append(anno)
    prog_bar.update()

mmcv.dump(results, output_pkl)
print(f'{output_pkl}---Finish!')
