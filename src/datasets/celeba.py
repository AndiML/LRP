"""Represents a module containing CelebA dataset."""

import os
import numpy
import pandas
import torch
import torchvision.transforms as transforms
from PIL import Image
from lrp.src.datasets.dataset import Dataset, DatasetData

class CelebaFolder:
    """
        A Dataset that:
        - Takes a pandas.DataFrame of (image_name, attr1, attr2, … attr40).
        - Looks up each image under the provided image dir.
        - Applies a torchvision transform to the PIL image.
        - Returns (image_tensor, label) where label is one binary attribute (e.g. 'Smiling'), or the full 40-vector if target_attr is None.
        """

    def __init__(self, img_dir: str, attrs_df: pandas.DataFrame, transform=None, target_attr: str = None):
        """
        Args:
            img_dir (str): Path to folder containing all JPEGs.
            attrs_df (pd.DataFrame): DataFrame with columns ["image_id", "5_o_Clock_Shadow", …, "Young"].
                The index is arbitrary, but column 0 must be “image_id” (filename).
            transform (callable, optional): A torchvision transform to apply to each image.
            target_attr (str, optional): If provided (e.g. "Smiling"), returns only that attribute as a binary label.
                                        If None, returns the full 40-element (-1, +1) vector as a torch.Tensor.
        """
        self.img_dir = img_dir
        self.attrs_df = attrs_df.reset_index(drop=True)
        self.transform = transform
        self.target_attr = target_attr

        # Ensures "image_id" is a column
        if "image_id" not in self.attrs_df.columns:
            raise ValueError("attrs_df must have a column named 'image_id' for the filename.")

        # If the user wants a single binary attribute, ensure it exists
        if self.target_attr and self.target_attr not in self.attrs_df.columns:
            raise ValueError(f"target_attr='{self.target_attr}' not in DataFrame columns.")

    def __len__(self):
        return len(self.attrs_df)

    def __getitem__(self, idx: int):
        row = self.attrs_df.iloc[idx]
        img_name = row["image_id"]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.target_attr:
            # Single‐attribute label. Mapping −1 → 0, +1 → 1
            raw_val = row[self.target_attr]
            label = 0 if raw_val == -1 else 1
            return img, label
        else:
            # Return full 40‐vector as a torch Tensor of (0, +1)
            attr_values = row.drop("image_id").astype(int).values
            vec = torch.tensor(attr_values, dtype=torch.float32)
            vec = torch.div(vec + 1, 2, rounding_mode="floor")
            return img, vec

class Celeba(Dataset):
    """
    A wrapper around TorchVision’s CelebA that expects you to have already manually placed all CelebA files under
    path/celeba/.
    """
    dataset_id = 'celeba'
    """Machine-readable ID that uniquely identifies this dataset."""

    def __init__(self, path, target_attr=None) -> None:
        """
        Args:
            path (str): Root directory under which you placed a “celeba/” folder:
                        ```
                        <path>/celeba/
                            img_align_celeba/               ← all JPEG files
                            list_attr_celeba.txt
                            identity_CelebA.txt
                            list_bbox_celeba.txt
                            list_landmarks_align_celeba.txt
                            list_eval_partition.txt
                        ```
        """
        super().__init__()

        self.path = path
        self.name = 'CelebA'
        self.only_one_target_label = False if target_attr is None else True
        custom_transforms = transforms.Compose([
            transforms.CenterCrop((160, 160)),
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


        train_ds, val_ds, test_ds = self.load_celeba_splits(transform=custom_transforms, target_attr=None)
        self._training_data = train_ds
        self._validation_data = val_ds
        self._test_data = test_ds
    

    def load_celeba_splits(
            self,
            transform=None,
            target_attr: str = None
        ):
            """
            Args:
                img_dir (str): Path to folder with images.
                transform (callable, optional): torchvision transforms to apply (e.g. Resize→Crop→ToTensor→Normalize)
                target_attr (str): The name of one attribute column, e.g. "Smiling". If None, returns full 40-vector.

            Returns:
                train_ds, val_ds, test_ds  (all three are `CelebaFolder` instances)
            """

            img_dir = os.path.join(self.path, "celeba")
            attrs_df = pandas.read_csv(os.path.join(img_dir, "list_attr_celeba.csv"))
            if "image_id" not in attrs_df.columns:
                first_col = attrs_df.columns[0]
                attrs_df.rename(columns={first_col: "image_id"}, inplace=True)
            # Each line in list_eval_partition.txt is: "<filename> <0|1|2>"
            #   0 → train, 1 → valid, 2 → test
            partitions = {"train": set(), "valid": set(), "test": set()}
            with open(os.path.join(img_dir, "list_eval_partition.csv"), "r") as f:
                # Skips the header line
                next(f)
                for line in f:
                    filename, split_idx = line.strip().split(",")
                    split_idx = int(split_idx)
                    if split_idx == 0:
                        partitions["train"].add(filename)
                    elif split_idx == 1:
                        partitions["valid"].add(filename)
                    elif split_idx == 2:
                        partitions["test"].add(filename)

            train_mask = attrs_df["image_id"].isin(partitions["train"])
            valid_mask = attrs_df["image_id"].isin(partitions["valid"])
            test_mask  = attrs_df["image_id"].isin(partitions["test"])

            train_df = attrs_df[train_mask].reset_index(drop=True)
            valid_df = attrs_df[valid_mask].reset_index(drop=True)
            test_df  = attrs_df[test_mask].reset_index(drop=True)

            image_location = os.path.join(img_dir,"img_align_celeba", "img_align_celeba")
            train_ds = CelebaFolder(img_dir=image_location, attrs_df=train_df, transform=transform, target_attr=target_attr)
            val_ds   = CelebaFolder(img_dir=image_location, attrs_df=valid_df, transform=transform, target_attr=target_attr)
            test_ds  = CelebaFolder(img_dir=image_location, attrs_df=test_df, transform=transform, target_attr=target_attr)

            return train_ds, val_ds, test_ds
    
    @property
    def training_data(self) -> DatasetData:
        """Gets the training data of the dataset.

        Returns:
            DatasetData: Returns the training data of the dataset.
        """

        return self._training_data

    @property
    def validation_data(self) -> DatasetData:
        """Gets the validation data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._validation_data

    @property
    def test_data(self) -> DatasetData:
        """Gets the test data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._test_data

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """
        return self.training_data[0][0].shape

    @property
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Returns:
            int: Returns the number of distinct classes.
        """
        return 2 if self.only_one_target_label else 40
