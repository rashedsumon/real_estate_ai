import os
import kagglehub

def download_dataset():
    """
    Automatically downloads the latest version of the target low-light exposure dataset.
    Returns the local path to the downloaded files.
    """
    print("Initializing auto-download via kagglehub...")
    try:
        # Download latest version of the specified dataset
        path = kagglehub.dataset_download("zara2099/low-light-image-enhancement-dataset")
        print("Path to dataset files:", path)
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_dataset()