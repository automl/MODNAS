# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .base_provider import DataProvider
from search_spaces.MobileNetV3.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

__all__ = ["ImagenetDataProvider"]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

class ImagenetDataProvider(DataProvider):
    DEFAULT_PATH = "/data/datasets/ImageNet/imagenet-pytorch"

    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=0.8,
        n_worker=32,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
    ):

        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            from search_spaces.MobileNetV3.utils.my_dataloader import MyDataLoader

            assert isinstance(self.image_size, list)
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.image_size)  # active resolution for test
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader
        train_transforms = self.build_train_transform(image_size=[128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224])
        train_dataset = self.train_dataset(train_transforms)
        valid_size = None
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)
            train_size = len(train_dataset) - valid_size
            dataset_train, dataset_val = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
            sampler_train = torch.utils.data.DistributedSampler(
                ConcatDataset(dataset_train,dataset_val),
                num_replicas=num_replicas,
                rank=rank,
                shuffle=True)

            self.train = torch.utils.data.DataLoader(
                ConcatDataset(dataset_train,dataset_val),
                batch_size=train_batch_size,
                sampler=sampler_train,
                num_workers=10,
                pin_memory=True,
            )
            '''self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=train_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )'''
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas, rank
                )
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    sampler=train_sampler,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            self.valid = None
        valid_transform = self.build_valid_transform(image_size=224)
        test_dataset = self.test_dataset(valid_transform)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=10,
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )

        #if self.valid is None:
        self.valid = self.test


        train_size = len(self.train) * self.train.batch_size
        test_size = len(self.test) * self.test.batch_size

        print(f"Dataloader summary: train-test = {train_size}-{test_size}")

    @staticmethod
    def name():
        return "imagenet"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/imagenet")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        from torch.utils.data import Subset
        import random
        train_dset = datasets.ImageFolder(self.train_path, _transforms)
        choices = np.random.choice([i for i in range(len(train_dset))],size=2000,replace=False)
        train_dset_subset = Subset(train_dset,choices)
        train_dset_subset.transform = train_dset.transform
        return train_dset

    def test_dataset(self, _transforms):
        from torch.utils.data import Subset
        import random
        test_dset = datasets.ImageFolder(self.valid_path, _transforms)
        choices = np.random.choice([i for i in range(len(test_dset))],size=1000,replace=False)
        test_dset_subset = Subset(test_dset,choices)
        test_dset_subset.transform = test_dset.transform
        return test_dset_subset

    @property
    def train_path(self):
        return os.path.join(self.DEFAULT_PATH, "train")

    @property
    def valid_path(self):
        return os.path.join(self.DEFAULT_PATH, "val")

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print(
                "Color jitter: %s, resize_scale: %s, img_size: %s"
                % (self.distort_color, self.resize_scale, image_size)
            )

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print(
                "Use MyRandomResizedCrop: %s, \t %s"
                % MyRandomResizedCrop.get_candidate_image_size(),
                "sync=%s, continuous=%s"
                % (
                    MyRandomResizedCrop.SYNC_DISTRIBUTED,
                    MyRandomResizedCrop.CONTINUOUS,
                ),
            )
        else:
            resize_transform_class = transforms.RandomResizedCrop

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        # color augmentation (optional)
        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )
        if color_transform is not None:
            train_transforms.append(color_transform)

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid, train and test dataset
        #self.train.dataset.transform = self.build_train_transform(
        #    image_size=self.active_img_size, print_log=False
        #)
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            #train_resolutions = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224]
            #train_transforms_factory = {}
            #for resolution in train_resolutions:
            #    train_transforms_factory[str(resolution)] = self.build_train_transform(resolution)
            train_transform = self.build_train_transform(224)
            new_train_dataset = self.train_dataset(
                train_transform)
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                #pin_memory=True,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]