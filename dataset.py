import os
import sys
import json
import albumentations as A
import time
import math

from tqdm import tqdm
from skimage import io
from transforms import *
from torch.utils.data import Dataset


def img_loader(dataset, num_classes, phase):
    cell_classes =['浆细胞','淋巴细胞','嗜酸性粒细胞','中性粒细胞']
    keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, num_classes)]
    img_dir, pnt_dir = f'./datasets/{dataset}/{phase}_image', f'./datasets/{dataset}/{phase}_point'
    data = []
    files = []
    reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    time_string = time.strftime('[%D-%H:%M:%S]',time.localtime())
    reader.set_description(f"{time_string} loading {phase} data")
    cnt =0
    for file in reader:
        #cnt += 1
        #if cnt > 10:
        #    return data, files
        files.append(file)
        values = [io.imread(os.path.join(img_dir, file))]
        with open(f"./datasets/{dataset}/{phase}_point/{os.path.splitext(file)[0]}.json", encoding='utf-8') as f:
            annotations = json.loads(f.read())
            #values += [np.array(annotations[c]).reshape(-1, 2) for c in annotations['classes']]
            values += [np.array(annotations[c]).reshape(-1, 2) for c in cell_classes]
            #print(values[1:])
            #print(values[1].shape,values[2].shape,values[3].shape,values[4].shape)
        data.append(dict(zip(keys, values)))

    return data, files


class DataFolder(Dataset):
    def __init__(self, dataset, num_classes, phase, data_transform):
        self.phase = phase
        self.data, self.files = img_loader(dataset, num_classes, phase)
        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'
        index = index % len(self.data)
        sample = self.data[index]
        sample = self.data_transform(sample)
        return sample


def build_dataset(args, image_set):
    mean, std = np.load(f'./datasets/{args.dataset}/mean_std.npy')
    additional_targets = {}
    for i in range(1, args.num_classes):
        additional_targets.update({'keypoints%d' % i: 'keypoints'})
    if image_set == 'train':
        augmentor = A.Compose([
            A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0, border_mode=0, value=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomCrop(height=1080, width=1080, always_apply=True),
            # A.RandomCrop(height=1024, width=1024, always_apply=True),
        ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    elif image_set == 'test':
        transform = Preprocessing(mean, std)
    else:
        raise NotImplementedError
    data_folder = DataFolder(args.dataset, args.num_classes, image_set, transform)
    return data_folder
