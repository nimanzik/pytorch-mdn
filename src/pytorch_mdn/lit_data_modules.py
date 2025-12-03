from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
import torch
from lightning import LightningDataModule
from sklearn import set_config as sklearn_set_config
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, Subset, TensorDataset

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from torch.utils.data import Dataset as TorchDataset


sklearn_set_config(transform_output="polars")


class TabularDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for tabular data using Polars DataFrames.

    This module handles splitting the data into train/validation/test sets
    and creates appropriate DataLoaders for each. It accepts Polars
    DataFrames for features, labels, and optional metadata.

    When random_split is True, the data is shuffled once using the
    specified seed before being split deterministically into
    train/validation/test sets. This ensures reproducible splits.

    The feature and label transformers are fitted on the training data and
    applied to all splits (train/validation/test). This ensures that
    validation and test data are scaled consistently with the training data.

    Parameters
    ----------
    x_df : pl.DataFrame
        Features DataFrame.
    y_df : pl.DataFrame
        Labels DataFrame.
    meta_df : pl.DataFrame or None, default=None
        Optional metadata DataFrame. Currently not used.
    x_scaler : FunctionTransformer or MinMaxScaler or StandardScaler \
or None, default=None
        Transformer for features. If None, uses FunctionTransformer
        (identity transformation).
    y_scaler : FunctionTransformer or MinMaxScaler or StandardScaler \
or None, default=None
        Transformer for labels. If None, uses FunctionTransformer
        (identity transformation).
    val_ratio : float, default=0.2
        Validation split ratio between 0 and 1.
    test_ratio : float, default=0.0
        Test split ratio between 0 and 1.
    random_split : bool or {"train_val"}, default=False
        Data splitting strategy:
        - False: No shuffling, data stays in original order
        - True: Entire dataset shuffled before splitting
        - "train_val": Test data taken from end in order, train-val shuffled
    seed : int or None, default=None
        Random seed for reproducibility. Only used when random_split is
        True. If None, shuffling is non-deterministic.
    batch_size : int, default=32
        Batch size for data loaders.
    num_workers : int, default=0
        Number of workers for data loading.
    shuffle_train : bool, default=True
        Whether to shuffle training data in each epoch. This is independent
        of random_split.
    pin_memory : bool, default=True
        Whether to pin memory in data loaders for faster GPU transfer.
    """

    def __init__(
        self,
        x_df: pl.DataFrame,
        y_df: pl.DataFrame,
        meta_df: pl.DataFrame | None = None,
        x_scaler: FunctionTransformer | MinMaxScaler | StandardScaler | None = None,
        y_scaler: FunctionTransformer | MinMaxScaler | StandardScaler | None = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        random_split: bool | Literal["train_val"] = False,
        seed: int | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        # Save all the hyperparameters passed to the constructor
        self.save_hyperparameters(
            ignore=["x_df", "y_df", "meta_df", "x_scaler", "y_scaler"]
        )

        self.x_df = x_df
        self.y_df = y_df
        self.meta_df = meta_df

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_split = random_split
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory

        self.x_scaler = x_scaler or FunctionTransformer()
        self.y_scaler = y_scaler or FunctionTransformer()
        self.x_scaler.set_output(transform="polars")
        self.y_scaler.set_output(transform="polars")

        self.train_dataset: TorchDataset[Any] | None = None
        self.val_dataset: TorchDataset[Any] | None = None
        self.test_dataset: TorchDataset[Any] | None = None

    @property
    def input_dim(self) -> int:
        """Get the input feature dimension."""
        return self.x_df.width

    @property
    def output_dim(self) -> int:
        """Get the output label dimension."""
        return self.y_df.width

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for each split.

        Converts Polars DataFrames to PyTorch tensors and splits the data
        into train/validation/test sets. If random_split is True, the data
        is shuffled once using the seed before being split deterministically.
        The splits are created using index ranges to ensure no additional
        shuffling occurs.

        The feature and label transformers are fitted on the training split
        and then applied to all splits to ensure consistent scaling.

        Parameters
        ----------
        stage : str or None, default=None
            Training stage ('fit', 'validate', 'test', 'predict'), or None.
        """
        # Only split once
        if self.train_dataset is not None:
            return

        # Determine split sizes
        n_samples = len(self.x_df)
        n_test = int(round(n_samples * self.test_ratio))
        n_val = int(round(n_samples * self.val_ratio))
        n_train = n_samples - (n_val + n_test)

        # Create index ranges for each split
        train_end = n_train
        val_end = n_train + n_val

        # Apply splitting strategy based on random_split parameter
        if self.random_split is True:
            # Full shuffle
            generator = (
                None if self.seed is None else torch.Generator().manual_seed(self.seed)
            )
            indices = torch.randperm(n_samples, generator=generator).tolist()
        elif self.random_split == "train_val":
            # Test data from end, shuffle train+val data
            generator = (
                None if self.seed is None else torch.Generator().manual_seed(self.seed)
            )

            # Test indices from end (in order)
            test_start_idx = n_samples - n_test if n_test > 0 else n_samples
            test_idxs = list(range(test_start_idx, n_samples))

            # Shuffle train+val indices
            train_val_indices = (
                torch.randperm(test_start_idx, generator=generator).tolist()
                if test_start_idx > 0
                else []
            )

            # Combine for final indices array
            indices = train_val_indices + test_idxs
        else:  # random_split is False
            # No shuffling at all - keep original order
            indices = list(range(n_samples))

        # Split indices
        train_idxs = indices[:train_end]
        val_idxs = indices[train_end:val_end] if n_val > 0 else []
        test_idxs = indices[val_end:] if n_test > 0 else []

        # Fit transformers on training data only
        self.x_scaler.fit(self.x_df[train_idxs])
        self.y_scaler.fit(self.y_df[train_idxs])

        # Transform all data
        x_df_trans: pl.DataFrame = self.x_scaler.transform(self.x_df)
        y_df_trans: pl.DataFrame = self.y_scaler.transform(self.y_df)

        # Create full tensor dataset
        full_dataset = TensorDataset(
            x_df_trans.to_torch(dtype=pl.Float32),
            y_df_trans.to_torch(dtype=pl.Float32),
        )

        # Create datasets using the split indices
        self.train_dataset = Subset(full_dataset, train_idxs)
        if val_idxs:
            self.val_dataset = Subset(full_dataset, val_idxs)
        if test_idxs:
            self.test_dataset = Subset(full_dataset, test_idxs)

    def _create_data_loader(
        self, data_set: TorchDataset[Any], shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader for a given dataset, generic method."""
        return DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns
        -------
        dl : DataLoader
            Training data loader.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return self._create_data_loader(self.train_dataset, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader | None:
        """Create validation DataLoader.

        Returns
        -------
        dl : DataLoader or None
            Validation data loader, or None if validation split is 0.
        """
        if self.val_dataset is None:
            return None

        return self._create_data_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        """Create test DataLoader.

        Returns
        -------
        dl : DataLoader or None
            Test data loader, or None if test split is 0.
        """
        if self.test_dataset is None:
            return None

        return self._create_data_loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader using test data or training data.

        Returns
        -------
        dl : DataLoader
            Prediction data loader.
        """
        # Use test dataset if available, otherwise use training dataset
        dataset = (
            self.test_dataset if self.test_dataset is not None else self.train_dataset
        )

        if dataset is None:
            raise RuntimeError(
                "No dataset available for prediction. Call setup() first."
            )

        return self._create_data_loader(dataset, shuffle=False)

    def inverse_transform_y(
        self, y: torch.Tensor | NDArray | pl.DataFrame
    ) -> pl.DataFrame:
        """Inverse transform labels back to original scale.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted. Call setup() first.
        """
        check_is_fitted(
            self.y_scaler, msg="'y_scaler' is not fitted. Call setup() first."
        )

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()

        y_inv = self.y_scaler.inverse_transform(y)
        return pl.DataFrame(y_inv, schema=self.y_df.schema)

    def inverse_transform_x(
        self, x: torch.Tensor | NDArray | pl.DataFrame
    ) -> pl.DataFrame:
        """Inverse transform features back to original scale."""
        check_is_fitted(
            self.x_scaler, msg="'x_scaler' is not fitted. Call setup() first."
        )

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()

        x_inv = self.x_scaler.inverse_transform(x)
        return pl.DataFrame(x_inv, schema=self.x_df.schema)
