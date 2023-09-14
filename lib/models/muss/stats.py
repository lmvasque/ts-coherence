import glob
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize

from lib.config import Config

plt.style.use('tableau-colorblind10')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
SEED = 123

import seaborn as sns
from sklearn.utils import shuffle

data_rename = {
    "en_mined": "Mined",
    "en_d_wiki_best": "Mined+D-Wiki(best)",
    "en_d_wiki_last": "Mined+D-Wiki",
    "en_cochrane_best": "Mined+Cochrane(best)",
    "en_cochrane_last": "Mined+Cochrane",
    "d_wikipedia": "D-Wiki",
    "en_dwiki_cochrane_best": "Mined+D-Wiki+Cochrane(best)",
    "en_dwiki_cochrane_last": "Mined+D-Wiki+Cochrane",
    "cochrane": "Cochrane",
    "osec_short_article": "OneStopEn",
    "osec_all": "OneStopEn_all",
    "en_mined_wikilarge": "Mined+WikiLarge",
    "en_d_wiki_v2_last": "Mined+D-Wiki",
    "en_d_wiki_v2_best": "Mined+dWikiv2best"
}


def compare_output_sentence_count(key):
    config = Config()
    root = config.get_muss_root()

    models = []
    models_names = ["en_mined", "en_mined_cochrane_last", "en_mined_d_wiki_best", "en_mined_d_wiki_last",
                    "en_mined_cochrane_d_wiki_last"]

    for name in models_names:
        models.append("{}/ts_discourse/{}".format(root, name))

    for i, model in enumerate(models):
        pred_files = glob.glob("{}/*{}*test*pred".format(model, key))
        print(pred_files)

        for path in pred_files:

            if ".valid." in path:
                continue
            sentences_counts = []

            with open(path, 'r', encoding='utf-8') as f:
                for text in f:
                    sent = sent_tokenize(text)
                    sentences_counts.append(len(sent))
                print(min(sentences_counts))

                # plt.hist(sentences_counts)
                name = data_rename[Path(model).name]
                sns.distplot(sentences_counts, hist=True, kde=True,
                             kde_kws={'linewidth': 2},
                             label=name.replace("WL", "WikiLarge"),
                             color=colors[i])
            # plt.title("How many sentences are in the output? {}".format(Path(path).name.replace(".test.pred", "")))

    plt.title("Test Set: {}".format(data_rename[key]))
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("Number of Sentences")
    plt.xticks(range(0, max(sentences_counts) + 4, 2))
    plt.ylabel("Density")
    # plt.savefig("outputs_length_{}.svg".format(key))
    # plt.clf()
    plt.legend()
    plt.show()


def compare_sentences_ratio():
    model_path = "ts_discourse/en_mined"
    complex_files = glob.glob("{}/*test.complex".format(model_path))
    std_dev_ratios = []
    avg_ratios = []
    ratios_labels = []

    for complex_path in complex_files:
        pred_path = complex_path.replace(".complex", ".pred")
        ratios = []
        with open(complex_path, 'r') as f1, open(pred_path, 'r') as f2:
            for sent_c, sent_s in zip(f1, f2):
                ratio = calculate_ratio_by_word(sent_c, sent_s)
                ratios.append(ratio)

        std_dev_ratios.append(np.std(ratios))
        avg_ratios.append(np.average(ratios))
        ratios_labels.append(Path(pred_path).name.replace(".pred", ""))

    plt.boxplot(avg_ratios)
    # plt.bar(ratios_labels, avg_ratios, color=colors[1])
    # plt.bar(ratios_labels, std_dev_ratios, color=colors[0])

    plt.xticks(rotation=10)
    plt.title("What is the ratio between complex and simple sentences? {}".format(Path(model_path).name))
    plt.xlabel("Ratio")
    plt.ylabel("Tokens (Source - Target / Total) (%)")
    plt.show()


def calculate_ratio_by_sent_len(sent_c, sent_s):
    sent_a = sent_c.strip()
    sent_b = sent_s.strip()
    len_a = len(sent_tokenize(sent_a))
    len_b = len(sent_tokenize(sent_b))

    result = len_a - len_b
    return result


def compare_output_lenght_ratio_pred_ref(key):
    config = Config()
    root = config.get_muss_root()

    models = []
    models_names = ["en_mined", "en_mined_cochrane_last", "en_mined_d_wiki_best", "en_mined_d_wiki_last",
                    "en_mined_cochrane_d_wiki_last"]

    for name in models_names:
        models.append("{}/ts_discourse/{}".format(root, name))

    for i, model in enumerate(models):
        pred_files = glob.glob("{}/*{}*test*pred".format(model, key))
        pred_len = []
        ref_len = []
        # diff_len = []

        for pred_path in pred_files:
            dataset = Path(pred_path).name.replace(".test.pred", "")
            ref_path = config.get_ref_file(dataset, "test")
            with open(pred_path, 'r') as f1, open(ref_path, 'r') as f2:
                for sent_p, sent_r in zip(f1, f2):
                    a = len(sent_p)
                    b = len(sent_r)
                    pred_len.append(a)
                    ref_len.append(b)

            name = data_rename[Path(model).name]
            plt.scatter(ref_len, pred_len, label=name, alpha=0.5)
            plt.title("Test Set: {}".format(data_rename[key]))

    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("Reference Length")
    plt.ylabel("Prediction Length")

    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    plt.autoscale(False)
    plt.plot(yticks, yticks, color=colors[6])

    plt.legend()
    plt.show()


def compare_output_sentences_ratio_pred_ref(key, tag):
    config = Config()
    root = config.get_muss_root()
    plt.rcParams["figure.figsize"] = (18, 9)

    plt.rcParams.update({'font.size': 22,
                         'legend.fontsize': 14,
                         'axes.labelsize': 14,
                         'figure.figsize': (7.5, 5)})

    models = []
    models_names = ["en_mined", "en_cochrane_last", "en_d_wiki_v2_last", "en_dwiki_cochrane_last"]

    for name in models_names:
        models.append("{}/output/paper_v2/ts_discourse/{}".format(Path(root).parent, name))
        # models.append("{}/ts_discourse/{}".format(root, name))

    for i, model in enumerate(models):
        pred_files = glob.glob("{}/*{}*test*pred".format(model, key))
        pred_len = []
        ref_len = []

        ax = plt.gca()
        for pred_path in pred_files:
            pred_shorter_than_ref = 0
            total_count = 0
            dataset = Path(pred_path).name.replace(".test.pred", "")
            ref_path = config.get_ref_file(dataset, "test")
            with open(pred_path, 'r') as f1, open(ref_path, 'r') as f2:
                for sent_p, sent_r in zip(f1, f2):
                    if "words" in tag:
                        a = len(word_tokenize(sent_p))
                        b = len(word_tokenize(sent_r))
                    elif "sentences" in tag:
                        a = len(sent_tokenize(sent_p))
                        b = len(sent_tokenize(sent_r))

                    pred_len.append(a)
                    ref_len.append(b)

                    if a < b:
                        pred_shorter_than_ref = pred_shorter_than_ref + 1

                    total_count = total_count + 1

            print("Pred are shorter than ref? {} {} {}%".format(Path(model).name,
                                                                dataset,
                                                                round((pred_shorter_than_ref / total_count) * 100, 3)))

        name = data_rename[Path(model).name]

        ref_len_rand, pred_len_rand = shuffle(ref_len, pred_len, random_state=SEED)

        for x, y in zip(ref_len_rand, pred_len_rand):
            ax.scatter(x, y, label=name, alpha=0.6)

        if max(pred_len_rand) < max(ref_len_rand):
            max_element = max(pred_len_rand)
            axis = sorted(pred_len_rand)
        else:
            max_element = max(ref_len_rand)
            axis = sorted(ref_len_rand)

        axis = [int(a) for a in axis]
        axis = list(set(axis))
        axis.append(0)
        axis.append(max_element + int(max_element * 0.10))

        plt.plot(axis, axis, color=colors[5])
    # plt.xlabel("Reference ({})".format(tag))
    # plt.ylabel("Prediction ({})".format(tag))

    labels = [data_rename[model] for model in models_names]
    patches = []
    for color, name in zip(colors[0:5], labels):
        patches.append(mpatches.Patch(color=color, label=name))

    plt.legend(handles=patches)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.savefig("{}/output/paper_ipm/img/{}.{}.png".format(Path(root).parent, dataset, tag))
    plt.clf()


def calculate_ratio_by_char(sent_a, sent_b):
    # len_a = len(word_tokenize(sent_a))
    # len_b = len(word_tokenize(sent_b))
    len_a = len(sent_a)
    len_b = len(sent_b)
    result = (len_a - len_b) / len_a * 100
    return result


def calculate_ratio_by_word(sent_a, sent_b):
    sent_a = sent_a.strip()
    sent_b = sent_b.strip()
    len_a = len(word_tokenize(sent_a))
    len_b = len(word_tokenize(sent_b))

    result = (len_a - len_b) / len_a * 100
    return result


def get_datasets_stats():
    root = "data/d_wikipedia"

    for subset in ["test"]:
        results = []
        with open("{}/{}.complex".format(root, subset)) as f1, \
                open("{}/{}.simple".format(root, subset)) as f2, \
                open("{}/{}.complex.clean".format(root, subset), "w") as f3, \
                open("{}/{}.simple.clean".format(root, subset), "w") as f4:

            for complex_line, simple_line in zip(f1, f2):
                word_len = len(word_tokenize(complex_line))
                if word_len < 680:
                    f3.write(complex_line)
                    f4.write(simple_line)

                    results.append(word_len)

        sns.distplot(results, hist=False, label=subset)
        print(sorted(results))

    plt.show()


def get_datasets_table():
    latex = \
        "\\hline\n\\textbf{Dataset} & \\textbf{Split} & \\textbf{Samples} & \\textbf{Sent} & \\textbf{Sent/Doc} & \\textbf{Words}  & \\textbf{Words/Doc} \\\\\n\\hline \\hline\n"
    root = "data/"
    files = ["{}/devaraj_2021/train.source".format(root), "{}/devaraj_2021/val.source".format(root),
             "{}/devaraj_2021/test.source".format(root),
             "{}/d_wikipedia_clean/train.src".format(root), "{}/d_wikipedia_clean/valid.src".format(root),
             "{}/d_wikipedia_clean/test.src".format(root),
             "{}/asset_clean/asset.valid.orig".format(root), "{}/asset_clean/asset.test.orig".format(root),
             "{}/OneStopEnglishCorpus/Texts-SeparatedByReadingLevel/Adv-Txt/adv.src".format(root)]

    labels = ["Cochrane(tr)", "Cochrane(v)", "Cochrane(t)",
              "D-Wiki(tr)", "D-Wiki(v)", "D-Wiki(t)",
              "ASSET(v)", "ASSET(t)", "OneStopEN"]

    for subset, label in zip(files, labels):

        with open(subset) as f1:
            doc_count = 0
            word_count = 0
            sent_count = 0
            for line in f1:
                if len(line.strip()) != 0:
                    doc_count = doc_count + 1
                    word_count = word_count + len(word_tokenize(line))
                    sent_count = sent_count + len(sent_tokenize(line))
            sent_count_avg = sent_count / doc_count
            word_count_avg = word_count / sent_count

            latex = latex + (
                "{} & {} & \\numprint{{{}}} & \\numprint{{{}}} "
                "& {} & \\numprint{{{}}} & {} \\\\\n\hline\n".format(label, label,
                                                                     doc_count,
                                                                     sent_count,
                                                                     round(sent_count_avg, 2),
                                                                     word_count,
                                                                     round(word_count_avg, 2)))

    with open("table1_dataset_stats.txt", "w") as w1:
        w1.write(latex)


if __name__ == '__main__':
    get_datasets_table()
# compare_output_sentences_ratio_pred_ref("cochrane", "words")
# compare_output_sentences_ratio_pred_ref("cochrane", "sentences")
# # #
# compare_output_sentences_ratio_pred_ref("d_wikipedia", "words")
# compare_output_sentences_ratio_pred_ref("d_wikipedia", "sentences")
# #
# # # compare_output_sentences_ratio_pred_ref("osec_short_article", "words")
# # # compare_output_sentences_ratio_pred_ref("osec_short_article", "sentences")
# # #
# compare_output_sentences_ratio_pred_ref("osec_all", "words")
# compare_output_sentences_ratio_pred_ref("osec_all", "sentences")

# compare_output_sentences_ratio_pred_ref("d_wikipedia", [], [])
# compare_output_sentence_count("asset")
# compare_output_sentence_count("osec_sentence_aligned")
# compare_output_sentence_count("d_wikipedia")
# compare_output_sentence_count("cochrane")

# compare_output_sentences_ratio_pred_ref("asset", [], [])
# compare_output_sentences_ratio_pred_ref("osec_sentence_aligned", [], [])
# compare_output_sentences_ratio_pred_ref("osec_short_article", [], [])
# compare_output_sentences_ratio_pred_ref("d_wikipedia", [], [])
# compare_output_sentences_ratio_pred_ref("cochrane", [], [])

# compare_output_lenght_ratio_pred_ref("asset")
# compare_output_lenght_ratio_pred_ref("osec_sentence_aligned")
# compare_output_lenght_ratio_pred_ref("osec_short_article")
# compare_output_lenght_ratio_pred_ref("d_wikipedia")
# compare_output_lenght_ratio_pred_ref("cochrane")
# compare_sentences_ratio()
# get_datasets_stats()
