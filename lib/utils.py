import pandas as pd
from matplotlib import pyplot as plt
from nltk import sent_tokenize, word_tokenize


def read(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    return data


def save_data(data, output_file, new_line="\n"):
    if isinstance(data, pd.DataFrame):
        text = data.values
    else:
        text = data

    with open(output_file, 'w') as f:
        for line in text:
            f.write("{}{}".format(line, new_line))


def save_line(data, output_file):
    with open(output_file, 'w') as f:
        f.write("{}\n".format(data))


def save_subsets_data(data, output_file):
    # {"test": {"src": [], "dst": []}, "dev": {"src": [], "dst": []}, "train": {"src": [], "dst": []}}
    for name, subset in data.items():
        for suffix, file in subset.items():
            path = "{}.{}.{}".format(output_file, name, suffix)
            with open(path, 'w') as f:
                for line in data[name][suffix]:
                    f.write("{}\n".format(line))


def data_analysis():
    plt.rcParams.update({'font.size': 16,
                         'legend.fontsize': 13,
                         'axes.labelsize': 14,
                         'figure.figsize': (7.5, 5)})
    path_wiki = "Document-level-text-simplification/Dataset/train.src"
    path_cochrane = "data/devaraj_2021/train.source"

    for path, color, key in zip([path_wiki, path_cochrane], ["#377eb8", "#ff7f00"], ["D-Wikipedia", "Cochrane"]):
        df = pd.read_csv(path, sep="\n", names=["text"])

        df["sentences"] = df["text"].apply(lambda f: sent_tokenize(str(f)))
        df["sent_count"] = df["sentences"].apply(lambda f: len(f))

        df["words"] = df["text"].apply(lambda f: word_tokenize(str(f)))
        df["word_count"] = df["words"].apply(lambda f: len(f))
        df = df.sort_values(by="word_count")

        plt.scatter(x=df.sent_count, y=df.word_count, color=color, label=key, alpha=0.8)

    plt.legend()
    plt.xlabel("Sentences")
    plt.ylabel("Words")
    plt.legend()
    plt.savefig("img/coherence_train.png")


def find_common_lines(file_train, file_test):
    with open(file_train) as f1, open(file_test) as f2:
        train = [s.strip() for s in f1.readlines()]

        for line in f2:
            line = line.strip()

            if line in train:
                index = train.index(line)
                print("Line from valid in train! Index: {}".format(index))


def main():
    file_train = "Document-level-text-simplification/Dataset/train.src"
    file_valid = "Document-level-text-simplification/Dataset/valid.src"
    find_common_lines(file_train, file_valid)


main()
