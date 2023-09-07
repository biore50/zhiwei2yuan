import os

from PersistStorage import PersistStorage

import shutil

import os.path as osp
import sys

rawdatadir = "/home/lwy/projects/Dataset/CSL450/ucf/test"
datadir = "/home/lwy/projects/Dataset/CSL450/ske"
outrawdatadir = "/home/lwy/projects/Dataset/CSL450/split/test"
labeldir = "/home/lwy/projects/Dataset/CSL50/label_name.txt"


assert os.path.exists(rawdatadir)
assert os.path.exists(outrawdatadir)
assert os.path.exists(datadir)

file_Object = PersistStorage()

classes = os.listdir(rawdatadir)

# ## label write
# # f = open(labeldir, 'a')
# # for class_folder in classes:
# #     f.write('\n{}'.format(class_folder))
#
# with open(labeldir, "r", encoding="utf8") as f:
#     lines = f.readlines()
#     with open(labeldir, "w", encoding="utf8") as f1:
#         for i,j in enumerate(lines):   # i 表示行号，j 表示每行内容
#             f1.write("{} {}".format(i+1,j))   # 枚举默认从0开始，这里进行+1后就是按照从第一行为1进行写入




classes = os.listdir(rawdatadir)
for class_folder in classes:
    sample_path = os.path.join(rawdatadir, class_folder)
    folder = os.listdir(sample_path)
    for sample in folder:
        # class_name = class_folder.split('_')[1]
        # name = class_folder.split('.')[0]
        name = sample + '.json'
        output_path = os.path.join(outrawdatadir, class_folder)
        if not osp.exists(output_path):
            os.makedirs(output_path)
        video = os.path.join(datadir, name)
        # output_video = os.path.join(output_path, class_folder)
        shutil.move(video, output_path)




# for class_folder in classes:
#     class_name = class_folder.split('_')[1]
#     name = class_folder.split('.')[0]
#     output_path = os.path.join(outrawdatadir, class_name)
#     if not osp.exists(output_path):
#         os.makedirs(output_path)
#     video = os.path.join(rawdatadir, class_folder)
#     output_video = os.path.join(output_path, class_folder)
#     shutil.move(video, output_video)
