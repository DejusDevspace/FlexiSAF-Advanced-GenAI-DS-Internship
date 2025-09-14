import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic processing of the wine quality datasets."""

    def __init__(self):
        self.dataset_path = Path(__file__).resolve().parent.parent / "dataset"
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def combine_datasets(
        self,
        d1: pd.DataFrame,
        d2: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            combined_dataset = pd.concat([d1, d2], ignore_index=True)
            logger.info("Combined dataset shape:", combined_dataset.shape)

            return combined_dataset

        except Exception as e:
            logger.error("Error combining datasets:", str(e))
            raise

    def save_dataset(self, data: pd.DataFrame, filename: str) -> None:
        """Save a dataset to file."""
        try:
            file_path = self.dataset_path / f"{filename}.csv"
            data.to_csv(file_path, index=False)

            logger.info(f"Dataset saved to: '{file_path}'")

        except Exception as e:
            logger.error("Error saving dataset:", str(e))

