import pandas as pd
import re


def process_data(data_file, stop_words_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train datado a research of your own
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    with open(stop_words_file) as f:
        stop_words = re.split("[\n\t]", f.read().strip())

    # dataframe with columns text and author
    df = pd.read_csv(data_file)
    df = df.drop(columns=["Unnamed: 0", "id"])

    y = list(df["author"])
    for idx, autrhor in enumerate(y):
        y[idx] = autrhor.strip()

    X = []

    for idx, citation in enumerate(df["text"]):
        cur_citation_words = []

        for word in re.findall("\w+'?\w*", citation):
            word = word.lower()

            if word not in stop_words:
                cur_citation_words.append(word)

        X.append(" ".join(cur_citation_words))

    return X, y
