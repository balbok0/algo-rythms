from zipfile import ZipFile
import os
import shutil
import numpy as np
from tqdm import tqdm

# Assumes that the actual fma_full.zip is in external hard drive. Adjust this
# according the computer preferences
zip_location = "F:/fma_full.zip"


# Given the name of a file containing directory information for other files
# it will unzip the required files from zip_location in to a folder named ../data/unzipped
# Folder will have the same structure as parent zip file. If ../data/unzipped already
# exits it will be deleted.
# Set track_progress to True for progress bar.
def unzip_files(file_name, track_progress=False):
    # Clean if directory already exists
    if os.path.isdir("../data/unzipped"):
        shutil.rmtree("../data/unzipped")
    # Decompress required files
    files_to_unzip = np.load(file_name)
    with ZipFile(zip_location) as zipped:
        if not track_progress:
            for file in files_to_unzip:
                zipped.extract("fma_full/{}".format(file), "../data/unzipped/")
        else:
            print("unzipping {}".format(file_name))
            for file in tqdm(files_to_unzip):
                zipped.extract("fma_full/{}".format(file), "../data/unzipped/")