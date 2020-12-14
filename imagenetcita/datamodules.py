
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Dataset
from imagenetcita.imagenet_classes import labels_dict
import pytorch_lightning as pl

class ImageNetIDWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.name_map_dict = labels_dict
        self.labels = self.true_labels()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = self.labels[y].index

        return x, y

    def true_labels(self):
        classes = self.dataset.class_to_idx
        ret = [[] for _ in classes]

        for k, v in classes.items():
            ret[v] = self.name_map_dict[k]

        return ret


class ImagenetDataset(pl.LightningDataModule):
    def __init__(self, dl_path = "./DATA", batch_size = 64, num_workers=4, train_subpath="train", val_subpath="val"):
        super().__init__()
        self.batch_size = batch_size
        self.dl_path = dl_path
        self.num_workers = num_workers
        self.train_subpath = train_subpath
        self.val_subpath = val_subpath

    def prepare_data(self):
        """Download images and prepare images datasets."""

        if not self.data_path.is_dir():
            download_and_extract_archive(
                url=self.data_url,
                download_root=self.dl_path,
                remove_finished=True)

    def setup(self):

        # 2. Load the data + preprocessing & data augmentation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = ImageFolder(root=self.data_path / self.train_subpath,
                                    transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

        valid_dataset = ImageFolder(root=self.data_path / self.val_subpath,
                                    transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

        # Child class has to implement a .labels_dict
        self.train_dataset = ImageNetIDWrapper(train_dataset)
        self.valid_dataset = ImageNetIDWrapper(valid_dataset)


    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(
            dataset=_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        print('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        print('Validation data loaded.')
        return self.__dataloader(train=False)


class ImageWoofData(ImagenetDataset):
    def __init__(self, dl_path = "./DATA", batch_size = 64, num_workers=4):
        super().__init__(dl_path = dl_path, batch_size = batch_size, num_workers=num_workers)

        self.data_path = Path(self.dl_path) / 'imagewoof2-320'
        self.data_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"


class ImageNetteData(ImagenetDataset):
    def __init__(self, dl_path = "./DATA", batch_size = 64, num_workers=4):
        super().__init__(dl_path = dl_path, batch_size = batch_size, num_workers=num_workers)

        self.data_path = Path(self.dl_path) / 'imagenette2-320'
        self.data_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

class Imagenetcita(ImagenetDataset):
    def __init__(self, dl_path = "./DATA", batch_size = 64, num_workers=4):
        super().__init__(dl_path = dl_path, batch_size = batch_size, num_workers=num_workers, train_subpath="train", val_subpath="test")

        self.data_path = Path(self.dl_path) / 'petiteimagenet_300'
        self.data_url = "https://github.com/jspaezp/imagenetcita/releases/download/v0.0.1/petiteimagenet_300.tgz"