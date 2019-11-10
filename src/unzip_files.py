from zipfile import ZipFile
import os
import shutil
import numpy as np

zip_location = "F:/fma_full.zip"


# Given the name of a file containing directory information for other files
# this function will unzip the required files in to a folder named unzippped
# from the zipfile defined in zip_location.
# This folder will have the same structure as the zip file that the data is
# being extracted from. If there is already a folder called unzipped it will
# be deleted before extracting more files.
def unzip_files(file_name):
    if file_name is None:
        raise Exception("Genre is not defined")
    if os.path.isdir("F:/"):
        shutil.rmtree("../data/unzipped")
    files_to_unzip = np.load("data/processed_data/{}".format(file_name))
    with ZipFile(zip_location) as zipped:
        for file in files_to_unzip:
            zipped.extract(file, "../data/unzipped/")