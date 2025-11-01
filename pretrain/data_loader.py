#!/usr/bin/env python3
"""
data_loader.py - Data Loading for Discourse-Aware Pretraining

This class handles loading the preprocessed data from the checkpoints
and preparing it for the Hugging Face Trainer.
"""

import logging
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Union

# Import from the utils directory
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.checkpoint_manager import load_all_checkpoints, print_checkpoint_status
from baseline.logiqa_data import load_logiqa_dataset as load_logiqa_eval_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PretrainingDataLoader:
    """
    Loads preprocessed data from Parquet checkpoints and converts
    it into a Hugging Face Dataset for training.
    """

    def __init__(self, config):
        """
        Initialize the data loader with the project configuration.

        Args:
            config (Config): The project Config object.
        """
        self.config = config
        self.checkpoint_dir = config.CHECKPOINT_DIR
        logger.info(f"Initialized PretrainingDataLoader for directory: {self.checkpoint_dir}")

    def load_preprocessed_data_for_training(
        self,
        test_split_size: float = 0.01,
        seed: int = 42
    ) -> Optional[DatasetDict]:
        """
        Loads all preprocessed data from checkpoints, converts to a Hugging Face Dataset,
        and splits it into train/test sets.

        Args:
            test_split_size (float): The proportion of the dataset to reserve for testing/evaluation.
            seed (int): Random seed for the train/test split.

        Returns:
            DatasetDict: A dictionary containing 'train' and 'test' datasets,
                         or None if loading fails.
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING PREPROCESSED DATA FOR TRAINING")
        logger.info("="*80)

        try:
            # Load all checkpoint data into a single Pandas DataFrame
            logger.info(f"Loading all checkpoints from: {self.checkpoint_dir}")
            print_checkpoint_status(self.checkpoint_dir) # Show status
            
            all_data_df = load_all_checkpoints(self.checkpoint_dir)

            if all_data_df.empty:
                logger.error("✗ No data loaded. DataFrame is empty.")
                logger.info("💡 Please run 'preprocess_updated.py' first.")
                return None
            
            logger.info(f"✓ Successfully loaded {len(all_data_df):,} total documents.")

            # Convert Pandas DataFrame to Hugging Face Dataset
            logger.info("Converting Pandas DataFrame to Hugging Face Dataset...")
            full_dataset = Dataset.from_pandas(all_data_df)
            
            # Log the columns to ensure they match expectations
            logger.info(f"Dataset columns: {full_dataset.column_names}")
            
            # Ensure required columns are present
            required_cols = ['input_ids', 'attention_mask', 'connector_mask']
            if not all(col in full_dataset.column_names for col in required_cols):
                logger.warning(f"✗ Dataset is missing one or more required columns: {required_cols}")
                logger.info("Please re-run preprocessing.")
                return None

            # Create train/test split
            logger.info(f"Creating train/test split (Test size: {test_split_size*100:.1f}%)")
            dataset_dict = full_dataset.train_test_split(
                test_size=test_split_size,
                seed=seed,
                shuffle=True
            )

            logger.info(f"  Train set size: {len(dataset_dict['train']):,}")
            logger.info(f"  Test set size: {len(dataset_dict['test']):,}")
            logger.info("Data loading complete.")
            
            return dataset_dict

        except FileNotFoundError:
            logger.error(f"Error: No checkpoint files found in '{self.checkpoint_dir}'.")
            logger.info("Please run 'preprocess_updated.py' to generate preprocessed data.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            return None

    def load_evaluation_data(self, dataset_name: str = "logiqa") -> Optional[Dataset]:
        """
        Loads dataset.

        Args:
            dataset_name (str): The name of the evaluation dataset to load.
                                Currently supports "logiqa".

        Returns:
            Dataset: The loaded evaluation dataset, or None if loading fails.
        """
        logger.info(f"\nLoading evaluation dataset: {dataset_name}")
        if dataset_name == "logiqa":
            try:
                logiqa_data = load_logiqa_eval_dataset()
                
                if isinstance(logiqa_data, list):
                    logiqa_dataset = Dataset.from_list(logiqa_data)
                elif isinstance(logiqa_data, pd.DataFrame):
                    logiqa_dataset = Dataset.from_pandas(logiqa_data)
                else:
                    logiqa_dataset = logiqa_data
                
                logger.info(f"Loaded {len(logiqa_dataset):,} LogiQA evaluation examples.")
                return logiqa_dataset
            except Exception as e:
                logger.error(f"Failed to load LogiQA dataset: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"Evaluation dataset '{dataset_name}' is not supported.")
            return None

if __name__ == "__main__":
    try:
        from utils.config import Config
        
        config = Config()
        
        data_loader = PretrainingDataLoader(config)
        
        # Try loading the pretraining data
        training_data_dict = data_loader.load_preprocessed_data_for_training()
        
        if training_data_dict:
            logger.info("\n--- Pre-training Data (Sample) ---")
            logger.info(training_data_dict['train'][0])
        else:
            logger.info("\nSkipping pre-training data example (loading failed).")

        # Try loading the evaluation data
        eval_dataset = data_loader.load_evaluation_data("logiqa")
        if eval_dataset:
            logger.info("\n--- Evaluation Data (Sample) ---")
            logger.info(eval_dataset[0])
            
    except ImportError:
        logger.error("Could not import Config from utils.config. Run this from the project root.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
