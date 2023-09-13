import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, transforms=None, txt_name: str = "train.txt"):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "TP-Dataset")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        image_dir = os.path.join(data_root, 'JPEGImages')
        mask_dir = os.path.join(data_root, 'GroundTruth')

        txt_path = os.path.join(data_root, "Index", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), 'r+') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.mask = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images)) == len(self.mask)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.mask[idx]).convert('L')
        target = np.array(target) / 255
        mask = np.clip(target, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

