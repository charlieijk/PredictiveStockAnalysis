"""
Hugging Face Hub Integration for Model Management

This module provides utilities for uploading and managing models on Hugging Face Hub.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

logger = logging.getLogger(__name__)


class HuggingFaceModelManager:
    """
    Manager for uploading and downloading models from Hugging Face Hub.

    Features:
    - Upload trained models to HF Hub
    - Download models from HF Hub
    - Manage model versions and metadata
    - Support for multiple repo types (model, dataset, space)
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace model manager.

        Args:
            token: HuggingFace API token. If None, will try to get from HF_TOKEN env variable
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            logger.warning("No HF token provided. Some operations may fail.")

        self.api = HfApi(token=self.token)

        if self.token:
            try:
                login(token=self.token)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face Hub: {e}")

    def upload_model(
        self,
        model_path: str,
        repo_id: str,
        repo_type: str = "model",
        commit_message: Optional[str] = None,
        private: bool = False,
        create_if_not_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a model to Hugging Face Hub.

        Args:
            model_path: Path to the local model directory or file
            repo_id: Repository ID in format "username/repo-name"
            repo_type: Type of repo ("model", "dataset", or "space")
            commit_message: Custom commit message
            private: Whether to create a private repository
            create_if_not_exists: Create repo if it doesn't exist

        Returns:
            Dictionary with upload information
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Create repository if needed
        if create_if_not_exists:
            try:
                self.api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=private,
                    exist_ok=True
                )
                logger.info(f"Repository {repo_id} created or already exists")
            except Exception as e:
                logger.error(f"Failed to create repository: {e}")
                raise

        # Prepare commit message
        if commit_message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Upload model - {timestamp}"

        try:
            if model_path.is_dir():
                # Upload entire folder
                logger.info(f"Uploading folder {model_path} to {repo_id}")
                result = self.api.upload_folder(
                    folder_path=str(model_path),
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=commit_message
                )
            else:
                # Upload single file
                logger.info(f"Uploading file {model_path} to {repo_id}")
                result = self.api.upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=model_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=commit_message
                )

            logger.info(f"Successfully uploaded to {repo_id}")
            return {
                "success": True,
                "repo_id": repo_id,
                "commit_url": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def upload_trained_models(
        self,
        models_dir: str = "models",
        repo_id: str = "charlieijk/PredictiveStockAnalysis",
        private: bool = False
    ) -> Dict[str, Any]:
        """
        Upload all trained models from the models directory.

        Args:
            models_dir: Directory containing trained models
            repo_id: Repository ID
            private: Whether repository should be private

        Returns:
            Upload results
        """
        models_path = Path(models_dir)

        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_path}")

        # Create README for the model repo
        readme_content = self._generate_model_readme(models_path)
        readme_path = models_path / "README.md"

        with open(readme_path, "w") as f:
            f.write(readme_content)

        return self.upload_model(
            model_path=str(models_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload stock prediction models",
            private=private
        )

    def _generate_model_readme(self, models_path: Path) -> str:
        """Generate README.md content for model repository."""

        # Count model files
        pkl_files = list(models_path.glob("*.pkl"))
        keras_files = list(models_path.glob("*.keras"))
        h5_files = list(models_path.glob("*.h5"))

        readme = f"""---
tags:
- stock-prediction
- time-series
- machine-learning
- finance
license: mit
---

# Stock Prediction Models

This repository contains trained machine learning models for stock price prediction.

## Models Included

### Traditional ML Models ({len(pkl_files)} files)
"""

        for pkl_file in pkl_files:
            readme += f"- `{pkl_file.name}`\n"

        readme += f"""
### Deep Learning Models ({len(keras_files) + len(h5_files)} files)
"""

        for keras_file in keras_files:
            readme += f"- `{keras_file.name}`\n"

        for h5_file in h5_files:
            readme += f"- `{h5_file.name}`\n"

        readme += """
## Model Types

This repository includes the following model architectures:

1. **Linear Regression** - Baseline model with Ridge/Lasso variants
2. **Random Forest** - Ensemble of 100 decision trees
3. **Gradient Boosting** - Sequential ensemble model
4. **LSTM Neural Network** - Deep learning for time series (128→64→32 units)
5. **Ensemble Model** - Weighted combination of all models

## Features

The models are trained on 50+ engineered features including:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Moving averages (SMA, EMA)
- Volume indicators (OBV, Volume ROC)
- Price patterns and ratios
- Lagged and rolling window features

## Usage

```python
import joblib
from tensorflow import keras

# Load traditional ML model
model = joblib.load('random_forest_model.pkl')

# Load LSTM model
lstm_model = keras.models.load_model('lstm_model.keras')

# Make predictions
predictions = model.predict(X_test)
```

## Performance Metrics

Models are evaluated on:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

## Training Details

- Data split: 70% train, 15% validation, 15% test
- Cross-validation: Time series split
- Hyperparameter tuning: GridSearchCV
- Feature selection: Top 30 features by mutual information

## Repository

Full source code and documentation: [PredictiveStockAnalysis](https://github.com/charlieijk/PredictiveStockAnalysis)

## License

MIT License
"""

        return readme

    def download_model(
        self,
        repo_id: str,
        local_dir: str = "downloaded_models",
        repo_type: str = "model"
    ) -> Path:
        """
        Download a model from Hugging Face Hub.

        Args:
            repo_id: Repository ID
            local_dir: Local directory to download to
            repo_type: Type of repository

        Returns:
            Path to downloaded model
        """
        from huggingface_hub import snapshot_download

        try:
            logger.info(f"Downloading {repo_id} to {local_dir}")
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                token=self.token
            )
            logger.info(f"Successfully downloaded to {path}")
            return Path(path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def list_models(self, author: Optional[str] = None) -> list:
        """
        List models on Hugging Face Hub.

        Args:
            author: Filter by author (username)

        Returns:
            List of model information
        """
        try:
            models = self.api.list_models(author=author)
            return list(models)
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


def upload_to_huggingface(
    folder_path: str = "models",
    repo_id: str = "charlieijk/PredictiveStockAnalysis",
    token: Optional[str] = None,
    private: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to upload models to Hugging Face Hub.

    Args:
        folder_path: Path to folder containing models
        repo_id: Repository ID (format: "username/repo-name")
        token: HuggingFace token (if None, uses HF_TOKEN env var)
        private: Whether to make repository private

    Returns:
        Upload results dictionary

    Example:
        >>> result = upload_to_huggingface(
        ...     folder_path="models",
        ...     repo_id="charlieijk/SolarRegatta",
        ...     token=os.getenv("HF_TOKEN")
        ... )
    """
    manager = HuggingFaceModelManager(token=token)
    return manager.upload_trained_models(
        models_dir=folder_path,
        repo_id=repo_id,
        private=private
    )


def upload_via_git(
    repo_id: str,
    source_dir: str = "models",
    token: Optional[str] = None,
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload models to HuggingFace using Git (better for large files).

    Requires git-lfs to be installed:
        brew install git-lfs (macOS)
        git lfs install

    Args:
        repo_id: Repository ID (format: "username/repo-name")
        source_dir: Source directory with models
        token: HuggingFace token (if None, uses HF_TOKEN env var)
        commit_message: Custom commit message

    Returns:
        Upload results dictionary
    """
    import subprocess
    import shutil
    from pathlib import Path

    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env variable or pass token.")

    # Parse username and repo name
    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError("repo_id must be in format: username/repo-name")

    username, repo_name = parts
    repo_url = f"https://{username}:{token}@huggingface.co/{repo_id}.git"
    local_dir = Path("hf_models_temp")

    try:
        # Clone or pull repo
        if local_dir.exists():
            logger.info("Pulling latest changes...")
            subprocess.run(["git", "pull"], cwd=local_dir, check=True)
        else:
            logger.info(f"Cloning repository {repo_id}...")
            try:
                subprocess.run(["git", "clone", repo_url, str(local_dir)], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError:
                logger.info("Repository doesn't exist, creating new one...")
                local_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init"], cwd=local_dir, check=True)
                subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=local_dir, check=True)

        # Setup Git LFS
        gitattributes = local_dir / ".gitattributes"
        with open(gitattributes, 'w') as f:
            f.write("*.pkl filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.keras filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.h5 filter=lfs diff=lfs merge=lfs -text\n")

        # Copy models
        source = Path(source_dir)
        for pattern in ["*.pkl", "*.keras", "*.h5", "*.json"]:
            for file_path in source.glob(pattern):
                shutil.copy2(file_path, local_dir / file_path.name)
                logger.info(f"Copied: {file_path.name}")

        # Create README (use existing function)
        manager = HuggingFaceModelManager(token=token)
        readme_content = manager._generate_model_readme(local_dir)
        with open(local_dir / "README.md", 'w') as f:
            f.write(readme_content)

        # Commit and push
        if commit_message is None:
            commit_message = f"Upload models - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        subprocess.run(["git", "add", "."], cwd=local_dir, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=local_dir, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=local_dir, check=True)

        logger.info(f"✓ Successfully pushed to https://huggingface.co/{repo_id}")

        return {
            "success": True,
            "repo_id": repo_id,
            "method": "git",
            "url": f"https://huggingface.co/{repo_id}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Git upload failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Upload models
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        repo_id = "charlieijk/PredictiveStockAnalysis"

    method = sys.argv[2] if len(sys.argv) > 2 else "api"

    print(f"\n{'='*60}")
    print(f"Uploading models to {repo_id}")
    print(f"Method: {method}")
    print(f"{'='*60}\n")

    if method == "git":
        result = upload_via_git(repo_id=repo_id)
    else:
        result = upload_to_huggingface(repo_id=repo_id)

    if result["success"]:
        print(f"\n✓ Successfully uploaded!")
        print(f"  Repository: {result['repo_id']}")
        print(f"  URL: {result.get('commit_url') or result.get('url')}")
    else:
        print(f"\n✗ Upload failed: {result['error']}")
