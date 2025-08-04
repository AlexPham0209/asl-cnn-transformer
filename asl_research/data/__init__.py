import io
import os
import zipfile

import requests

URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
DATA_PATH = os.path.join("data", "raw")

if __name__ == "__main__":
    if not os.path.isdir(DATA_PATH):
        quit()

    print("Downloading dataset...")
    r = requests.get(URL, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_PATH)

    # Extract downloaded dataset
    print("Extracting Dataset...")
    with zipfile.ZipFile(os.path.join(DATA_PATH, "UCI HAR Dataset.zip"), "rb") as zip_ref:
        zip_ref.extractall(DATA_PATH)
