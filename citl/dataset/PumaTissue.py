import json
import os
import warnings

import albumentations as A
import geopandas as gpd
import numpy as np
import pytorch_lightning as L
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as F
from torchvision.transforms import v2

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class PumaTissueDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            tif_folder (str): Path to the folder containing .tif images.
            geojson_folder (str): Path to the folder containing the first set of .geojson files.
            transform (callable, optional): Optional transform to apply to the image and mask.
        """
        self.tif_folder = os.path.join(data_dir, "puma", "01_training_dataset_tif_ROIs")
        self.geojson_folder = os.path.join(
            data_dir, "puma", "01_training_dataset_geojson_tissue"
        )
        self.tif_files = sorted(os.listdir(self.tif_folder))
        self.geojson_files = sorted(os.listdir(self.geojson_folder))
        self.transform = transform
        self.augment_indices = {}
        self.augments = None

    def __len__(self):
        return len(self.tif_files)

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, idx):
        # Load .tif image
        tif_path = os.path.join(self.tif_folder, self.tif_files[idx])
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.uint8)

        # Load .geojson files and create masks
        geojson_path = os.path.join(self.geojson_folder, self.geojson_files[idx])

        with rasterio.open(tif_path) as src:
            transform = src.transform
            height, width = src.shape

        mask = self._geojson_to_mask(geojson_path, transform, height, width)

        if self.augment_indices[idx]:
            augmented = self.augments(image=img.transpose(1, 2, 0), mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)

        # Apply transformations
        if self.transform:
            img, mask = self.transform(img, mask)
        img = img[:3, :, :]
        return img, mask, idx

    @staticmethod
    def _geojson_to_mask(geojson_path, transform, height, width):
        """
        Convert a GeoJSON file to a binary mask.
        """
        gdf = gpd.read_file(geojson_path)
        mask = np.zeros((height, width), dtype=np.uint8)

        tissue_map = {
            "tissue_white_background": 0,
            "tissue_stroma": 1,
            "tissue_blood_vessel": 2,
            "tissue_tumor": 3,
            "tissue_epidermis": 4,
            "tissue_necrosis": 5,
        }

        for _, row in gdf.iterrows():
            shapes = [(row["geometry"], 1)]
            classification = json.loads(row["classification"])
            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=tissue_map[classification["name"]],
                all_touched=True,
                dtype=np.uint8,
            )
        return mask


class PumaTissueDataModule(L.LightningDataModule):
    classes = [
        "background",
        "stroma",
        "blood_vessel",
        "tumor",
        "epidermis",
        "necrosis",
    ]

    augments = None

    task = "segmentation"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size: int = 4,
        data_dir: str = PATH_DATASETS,
        noise_level: float = 0.0,
    ):
        super().__init__()

        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.num_classes = 10
        self.batch_size = batch_size

        self.image_size = 1024

        self.transform = v2.Compose(
            [
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Resize(
                    self.image_size, max_size=self.image_size + 1, antialias=False
                ),
                v2.CenterCrop(self.image_size),
            ]
        )

    def setup(self, stage=None):
        generator = torch.Generator()
        generator.manual_seed(42)

        puma_full = PumaTissueDataset(self.data_dir, transform=self.transform)
        train_size = 160
        val_size = 30
        test_size = 15
        self.puma_train, self.puma_val, self.puma_test = random_split(
            puma_full, [train_size, val_size, test_size]
        )
        puma_full.set_indices(
            self.puma_train.indices, self.puma_val.indices + self.puma_test.indices
        )
        puma_full.augments = self.augments

    def debug_dataloader(self):
        return DataLoader(
            self.puma_train,
            num_workers=0,
            shuffle=True,
            batch_size=self.batch_size,
            persistent_workers=False,
            drop_last=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.puma_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=self.batch_size,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.puma_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.puma_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )


__all__ = ["PumaTissueDataModule"]
