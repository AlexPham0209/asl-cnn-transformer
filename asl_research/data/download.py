import subprocess
import zipfile

import requests
from scipy import io

PHOENIX_LINK = "https://www.kaggle.com/api/v1/datasets/download/mariusschmidtmengin/phoenixweather2014t-3rd-attempt"


def main():
    print("Downloading Phoenix Dataset...")
    r = requests.get(URL, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_PATH)



if __name__ == "__main__":
    main()