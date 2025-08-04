import gzip
import os
import pickle
import pandas as pd

DATA_PATH = os.path.join('data', 'processed', 'phoenixweather2014t')

def preprocess_data(file_name: str, name: str):
    try:
        with gzip.open(os.path.join(DATA_PATH, file_name), 'rb') as f:
            annotations = pickle.load(f)
    except:
        print('Error: Invalid path')
        return

    VIDEO_PATH = os.path.join(DATA_PATH, 'videos_phoenix', 'videos')
    names = list(map(lambda x: os.path.join(VIDEO_PATH, *x.split('/')) + ".mp4", [key['name'] for key in annotations]))
    glosses = [key['gloss'] for key in annotations]
    texts = [key['text'] for key in annotations]

    print(f"Names Size: {len(names)}")
    print(f"Glosses Size: {len(glosses)}")
    print(f"Texts Size: {len(texts)}\n")

    data = {
        "names": names,
        "glosses": glosses,
        "texts": texts
    }

    df = pd.DataFrame(data)
    print(df.head(3))

    # Removing duplicate rows
    print(f"\nPercentage of Duplicated Data:\n{df.duplicated().sum() / len(df)}\n")
    df = df.drop_duplicates()

    # Removing rows with missing information
    print(f"\nPercentage of Missing Data:\n{df.isna().sum() / len(df)}\n")
    df = df.dropna()

    # Make text lowercase and glosses uppercase
    df['texts'] = df['texts'].str.lower()
    df['glosses'] = df['glosses'].str.upper()

    # Removing numbers
    df['texts'] = df['texts'].str.replace(r'\d+', '') 
    df['glosses'] = df['glosses'].str.replace(r'\d+', '') 

    # Filter by valid video path
    existing = df['names'].astype(str).map(lambda file: os.path.exists(file))
    df = df[existing]

    # Saving data frame as csv
    df.to_csv(os.path.join(DATA_PATH, name), index=False)  


# Preprocess the validation, training and testing sets 
preprocess_data("phoenix14t.pami0.train.annotations_only.gzip", "train.csv")
preprocess_data("phoenix14t.pami0.test.annotations_only.gzip", "test.csv")
preprocess_data("phoenix14t.pami0.dev.annotations_only.gzip", "dev.csv")
