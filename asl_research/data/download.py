import subprocess
import zipfile

import requests
import io
import os

PHOENIX_2014_T_LINK = "https://www.kaggle.com/api/v1/datasets/download/mariusschmidtmengin/phoenixweather2014t-3rd-attempt"
EXTERNAL_DATA_PATH = os.path.join("data", "external")
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def main():
    # Creating phoenix dataset directory
    try:
        os.mkdir(os.path.join(EXTERNAL_DATA_PATH, "phoenixweather2014t"))
        os.mkdir(os.path.join(PROCESSED_DATA_PATH, "phoenixweather2014t"))
        print(f"Directory '{"phoenixweather2014t"}' created successfully.")
    except FileExistsError:
        print(f"Directory '{"phoenixweather2014t"}' already exists.")
    
    # Downloading from the link and unzipping in both the processed and external directory
    print("Downloading Phoenix Dataset...")
    r = requests.get(PHOENIX_2014_T_LINK, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    print("Extracting contents...")
    z.extractall(os.path.join(EXTERNAL_DATA_PATH, "phoenixweather2014t"))
    z.extractall(os.path.join(PROCESSED_DATA_PATH, "phoenixweather2014t"))



if __name__ == "__main__":
    main()