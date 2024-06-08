import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import pandas as pd


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


mean = [0.2652, 0.2652, 0.2652]
std = [0.1994, 0.1994, 0.1994]
data_transforms = {
    'training': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(), ]), p=0.3),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3), ]), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class MammoDataset(Dataset):
    def __init__(self,
                 data_path,
                 metadata,
                 phase,
                 trasnform=None,
                 certain=True,
                 image_size = 224,
                 mask_patch_size = 32,
                 model_patch_size = 16,
                 mask_ratio = 0.6,
                 seed=None):
        self.phase = phase
        self.certain = certain
        self.data_path = data_path

        if (seed):
            seed_everything(seed)

        self.transform = data_transforms[self.phase] if (trasnform is None) else trasnform
        data = pd.read_csv(metadata)
        self.data = data.loc[data["split"] == phase].reset_index()
        self.mask_generator = MaskGenerator(
            input_size=image_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

    def get_path(self, data, index):
        image_name = data['image_id'].iloc[index]
        study_id = data['study_id'].iloc[index]
        image_path = os.path.join(self.data_path, study_id + '/' + image_name + '.png')
        return image_path

    def get_score(self, data, index):
        birads = data['breast_birads'].iloc[index]
        score = eval(birads[-1])
        return score

    def __getitem__(self, index):
        image_path = self.get_path(self.data, index)
        print(image_path)
        image = cv2.imread(image_path)
        image = self.transform(image)
        mask = self.mask_generator()
        score = self.get_score(self.data, index)
        return image, mask, score


if __name__ == "__main__":
    train_dataset = MammoDataset(data_path="/media/jackson/Data/archive/Processed_Images",
                                 metadata="/media/jackson/Data/archive/split_data.csv",
                                 phase="training",
                                 seed=1)

    print(train_dataset[0])
