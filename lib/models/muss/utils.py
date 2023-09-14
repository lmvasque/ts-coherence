import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize

FILE = "cochrane_experiments/cochrane_word_counts_per_line_all.txt"
ONESTOP_FILE = "cochrane_experiments/onestop_counts_per_line.txt"
ONESTOP_HALF_FILE = "cochrane_experiments/onestop_counts_per_line_half.txt"
DATA_CMD = "cat * | awk '{print NF}' | sort -n | uniq > cochrane_word_counts_per_line.txt"


def histogram_from_file(file):
    data = np.loadtxt(file)
    plt.hist(data, bins='auto')
    plt.title("Cochrane - Words per line")
    plt.show()


def histogram_from_file2(file):
    data = np.loadtxt(file)
    plt.hist(data, bins='auto')
    plt.title("OneStopEnglish - Words per line")
    plt.show()


def summarize_results():
    experiments = {'BART + mined': "local_1637146329604/checkpoints",
                   'BART + Wikilarge + mined': "local_1637277213149/checkpoints"}
    working_dir = "muss/experiments/fairseq/{}"
    resources_dir = "muss/resources/datasets"

    easse_cmd = "easse evaluate --orig_sents_path {} --test_set custom -i {} --refs_sents_paths {}"
    datasets = ['cochrane', 'asset']
    evalsets = ['valid', 'test']

    with open("eval_script.sh", "w+") as f:

        for name, experiment in experiments.items():
            exp_dir = working_dir.format(experiment)
            f.write("echo \"\\n=== Analyzing experiment [{}] ===\"\n".format(experiment))
            for dataset in datasets:
                simple_suffix = "simple"
                if "asset" in dataset:
                    simple_suffix = "simple.0"
                for subset in evalsets:
                    result = easse_cmd.format("{}/{}.{}.complex".format(exp_dir, dataset, subset),
                                              "{}/{}.{}.pred".format(exp_dir, dataset, subset),
                                              "{}/{}/{}.{}".format(resources_dir, dataset, subset, simple_suffix))
                    f.write("echo \"Name: {} Dataset: {} Set: {}\"\n".format(name, dataset, subset))
                    f.write("{}\n".format(result))


def prepare_one_stop_data():
    stats_x = []
    stats_y = []
    data_path = "OneStopEnglishCorpus/Texts-SeparatedByReadingLevel/{}"
    output_path = "ts-discourse/data/osec_all"

    ele_files_dir = data_path.format("Ele-Txt")
    adv_files_dir = data_path.format("Adv-Txt")

    ele_out_file = "{}/test.simple".format(output_path)
    adv_out_file = "{}/test.complex".format(output_path)

    if Path(ele_out_file).exists():
        os.remove(ele_out_file)

    if Path(adv_out_file).exists():
        os.remove(adv_out_file)

    samples = 0
    for ele_file in glob.glob("{}/*.txt".format(ele_files_dir)):
        print(ele_file)
        article_name = Path(ele_file).name
        adv_file = "{}/{}".format(adv_files_dir, article_name.replace('-ele', '-adv'))

        with open(ele_file, 'r') as f1, open(adv_file, 'r') as f2, \
                open(ele_out_file, 'a+') as f3, open(adv_out_file, 'a+') as f4:
            data_ele = f1.read()
            data_adv = f2.read()
            tokens_ele = len(word_tokenize(data_ele))
            tokens_adv = len(word_tokenize(data_adv))

            stats_x.append(tokens_ele)
            stats_y.append(tokens_adv)

            TR = 850
            if tokens_ele < TR and tokens_adv < TR:
                samples = samples + 1
                f3.write("{}\n".format("".join(data_ele.replace("\n", " "))))
                f4.write("{}\n".format("".join(data_adv.replace("\n", " "))))

    plt.scatter(stats_x, stats_y)
    plt.plot(stats_x, [TR] * len(stats_x))
    plt.plot([TR] * len(stats_x), stats_y)
    # plt.plot(stats_y, stats_y)
    plt.title("# Samples: {}".format(samples))
    plt.savefig("{}/samples_stats.png".format(output_path))
    plt.show()


def main():
    pass
    # summarize_results()
    # prepare_one_stop_data()
    # histogram_from_file2(ONESTOP_HALF_FILE)
    # histogram_from_file2(ONESTOP_FILE)
    # histogram_from_file(FILE)


main()
