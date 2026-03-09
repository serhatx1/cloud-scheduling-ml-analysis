import kagglehub
import shutil
import os

def download():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("zoya77/cloud-workload-dataset-for-scheduling-analysis")
    print("Path to dataset files:", path)
    
    # Move the file to the current directory for easier access
    filename = "cloud_workload_dataset.csv"
    src = os.path.join(path, filename)
    dest = os.path.join(os.getcwd(), filename)
    
    if os.path.exists(src):
        shutil.copy(src, dest)
        print(f"Copied {filename} to {dest}")
    else:
        print(f"Could not find {filename} in {path}")
        # List files to see what's there
        print("Files in path:", os.listdir(path))

if __name__ == "__main__":
    download()
