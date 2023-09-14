import glob
import os
import re
import subprocess
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import sent_tokenize
from textstat import textstat

from lib.config import Config

plt.style.use('tableau-colorblind10')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ["o", "^", "s", "*", "x", "H", "X", "d", ">", "<"]

# Step by step MUSS fine-tuning

# Step 0: Get all params
config = Config()
models = config.get_models()
root = config.get_muss_root()
evaluation_sets = config.get_eval_sets()
model_alias = config.get_model_alias()


def evaluate_models(name, path, complex_sent):
    os.chdir(root)
    cmd = "python scripts/train_model.py --evaluate --model_name {} --model_path {} --complex_sent {}".format(name,
                                                                                                              path,
                                                                                                              complex_sent)
    print(cmd)
    exit_code = os.system(cmd)
    print("Exit Code: {}".format(exit_code))
    return 0


def setup_dirs():
    os.chdir(root)
    for name, model_path in models:
        output_dir = "{}/ts_discourse/{}".format(root, name)
        dst_model_path = "{}/model.pt".format(output_dir)
        src_model_dir = os.path.dirname(model_path)
        dst_model_dir = os.path.dirname(dst_model_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.symlink(model_path, dst_model_path)
            os.symlink("{}/dict.complex.txt".format(src_model_dir), "{}/dict.complex.txt".format(dst_model_dir))
            os.symlink("{}/dict.simple.txt".format(src_model_dir), "{}/dict.simple.txt".format(dst_model_dir))

        evaluate_models(name, dst_model_dir, config.get_complex_sentences())


def run_easse():
    for name, model_path in models:
        output_dir = "{}/ts_discourse/{}".format(root, name)
        easse_report = "{}/easse_report.out".format(output_dir)

        if os.path.exists(easse_report):
            os.remove(easse_report)
        complex_files = glob.glob("{}/*.complex".format(output_dir))
        # complex_files = list(filter(lambda sent: "sentence_aligned" in sent, complex_files))
        # print(complex_files)

        for file in complex_files:
            test_set_name = os.path.basename(file)
            original_sent = "{}/{}".format(output_dir, test_set_name)
            system_sent = "{}/{}".format(output_dir, test_set_name.replace("complex", "pred"))
            dataset, subset, _ = test_set_name.split(".")
            ref_sent = config.get_ref_file(dataset, subset)
            cmd = "{} evaluate --orig_sents_path {} --test_set custom -i {} --refs_sents_paths {}".format(
                config.get_easse_cmd(), original_sent, system_sent, ref_sent)
            print(cmd)
            result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout)
            print(result.stderr)
            with open(easse_report, 'a+') as f:
                bleu, sari, fkgl = re.match(".*'bleu': (\\d+.\\d+), 'sari': (\\d+.\\d+), 'fkgl': (\\d+.\\d+).*",
                                            str(result.stdout)).groups()
                f.write("{}\n".format(",".join([name, test_set_name, bleu, sari, fkgl])))

    summary_file = "{}/ts_discourse/ts_discourse.csv".format(root)
    os.system("echo model,dataset,bleu,sari,fkgl > {}".format(summary_file))
    sum_cmd = "cat {}/ts_discourse/*/*easse* >> {}".format(root, summary_file)
    print(sum_cmd)
    os.system(sum_cmd)


def show_results(latex=False):
    report_file = "{}/ts_discourse/ts_discourse.csv".format(root)
    output_dir = os.path.dirname(report_file)
    df = pd.read_csv(report_file)
    df = df.convert_dtypes()
    df = df[df["dataset"].str.contains("test")]

    df['alias'] = df['model'].apply(lambda f: model_alias[f])
    plt.rcParams["figure.figsize"] = (18, 9)
    plt.rcParams.update({'font.size': 18})

    print(evaluation_sets)
    summary_show = df.copy()

    for key in ["sari", "fkgl", "bleu"]:
        summary_show[key] = summary_show[key].apply(lambda f: round(f, 3))

    summary_show["test_alias"] = summary_show["dataset"].apply(lambda s: data_rename[s.replace(".test.complex", "")])

    summary_show = summary_show.sort_values(by="sari", ascending=False)
    summary_show = summary_show.sort_values(by="alias", ascending=False)
    df_ready = summary_show[["alias", "test_alias", "sari", "fkgl", "bleu"]]

    if latex:
        df_ready["latex"] = df_ready['alias'] + " & " + df_ready['test_alias'] + " & " + df_ready['sari'].astype(str) + \
                            " & " + df_ready['fkgl'].astype(str) + " & " + df_ready['bleu'].astype(
            str) + " \\\\ \n\hline"
        result = "\n".join(df_ready["latex"].values).replace("_", "-")
        result = result.replace("WL", "WikiLarge")
        print(result)

    for score_type in ["sari", "bleu", "fkgl"]:
        for index, subset in enumerate(evaluation_sets):
            summary = show_line_plot(df, subset, score_type, index)

        plt.savefig("{}/ts_discourse_{}.png".format(output_dir, score_type))
        plt.clf()


def show_results_d_sari_fkgl(latex=False):
    easse_report_file = "ts_discourse.csv"
    sari_report_file = "d_sari.csv"
    output_dir = os.path.dirname(easse_report_file)

    df_fkgl = pd.read_csv(easse_report_file)
    df_fkgl = df_fkgl.convert_dtypes()
    df_fkgl = df_fkgl[df_fkgl["dataset"].str.contains("test")]
    df_fkgl = df_fkgl.drop(columns=["sari", "bleu"])

    df_sari = pd.read_csv(sari_report_file)
    df_sari = df_sari.convert_dtypes()
    df_sari = df_sari[df_sari["dataset"].str.contains("test")]
    df_sari = df_sari.rename(columns={"sari": "dsari"})

    summary = pd.merge(df_fkgl, df_sari, on=["model", "dataset"])
    summary = summary[~summary["model"].str.contains("best")]
    summary["fkgl"] = summary["fkgl"].apply(lambda f: round(f, 3))
    summary[["dsari", "keep", "delete", "add"]] = summary[["dsari", "keep", "delete", "add"]].apply(
        lambda f: round(f * 100, 3))
    summary['alias'] = summary['model'].apply(lambda f: model_alias[f])
    summary["test_alias"] = summary["dataset"].apply(lambda s: data_rename[s.replace(".test.complex", "")])
    summary = summary.sort_values(by="fkgl", ascending=False)
    summary = summary.sort_values(by="alias", ascending=False)
    summary["alias"] = summary["alias"].apply(lambda s: s.replace("(last)", ""))
    df_ready = summary[["alias", "test_alias", "fkgl", "dsari", "keep", "delete", "add"]]
    print(df_ready)

    if latex:
        latex_out = ""
        for index, line in df_ready.iterrows():
            latex_out = latex_out + " & ".join([str(s) for s in line.values]) + " \\\\ \n\\hline\n"
        print(latex_out)
        with open("{}/ts_discourse.latex".format(output_dir), "w") as f1:
            f1.write(latex_out)


def show_line_plot(df, subset, score_type, index):
    ax = plt.gca()
    marker_size = 120

    df_test = df[(df["dataset"].str.contains("test")) & (df["dataset"].str.contains(subset))]
    subset = subset.replace("osec_sentence_aligned", "onestop_sent")
    subset = subset.replace("osec_short_article", "onestop_art_3")

    df_test.plot.scatter(x="alias", y=score_type, ax=ax, label="{}.test".format(subset), marker=markers[index],
                         color=colors[index], s=marker_size)

    df_valid = df[df["dataset"].str.contains("valid") & df["dataset"].str.contains(subset)]
    if len(df_valid):
        df_valid.plot.scatter(x="alias", y=score_type, ax=ax, label="{}.valid".format(subset), marker=markers[index],
                              color=colors[index], s=marker_size)

    plt.xticks(rotation=8)
    plt.ylabel("{} Score".format(score_type.upper()))
    plt.xlabel("")

    return df_test


def count_lines_no_stop():
    file = "cochrane/train.complex"

    count = 0
    with open(file, "r") as f:
        for line in f:
            if not re.match('.*\\.$', line):
                print(line)
                count = count + 1

    print(count)


def get_sentence_aligned_dataset():
    adv_ele = "OneStopEnglishCorpus/Sentence-Aligned/ADV-ELE.txt"
    dest_path = "muss/resources/datasets/onestopenglish/sentence_aligned"
    adv_data = []
    ele_data = []

    count = 0
    with open(adv_ele, "r") as f:
        for line in f:
            if count == 0:
                adv_data.append(line)
                count = count + 1
            elif count == 1:
                ele_data.append(line)
                count = count + 1
            else:
                count = 0

    with open("{}/test.complex".format(dest_path), "w+") as f:
        for line in adv_data:
            f.write(line)

    with open("{}/test.simple".format(dest_path), "w+") as f:
        for line in ele_data:
            f.write(line)


def get_articles_words_count(file):
    words_count = {}
    cmd = "wc -w {}/*adv.txt > adv_files.txt".format(file)
    exit_code = os.system(cmd)
    print("Running: {}.\nExit Code: {}".format(cmd, exit_code))

    with open("adv_files.txt", "r") as f:
        for line in f:
            groups = line.strip().split(" ", 1)
            count = groups[0]
            file = groups[1]
            words_count[Path(file).name.strip()] = int(count)

    del (words_count["total"])
    return words_count


def filter_one_stop_by_tokens():
    data_path = "OneStopEnglishCorpus/Texts-SeparatedByReadingLevel/{}"
    adv_articles_wc = get_articles_words_count(data_path.format("Adv-Txt"))

    ele_files_dir = data_path.format("Ele-Txt")
    adv_files_dir = data_path.format("Adv-Txt")

    ele_out_file = "{}/test.simple".format(ele_files_dir)
    adv_out_file = "{}/test.complex".format(adv_files_dir)

    if Path(ele_out_file).exists():
        os.remove(ele_out_file)

    if Path(adv_out_file).exists():
        os.remove(adv_out_file)

    for name, count in adv_articles_wc.items():
        # if count < 1024:
        adv_file = "{}/{}".format(adv_files_dir, name)
        ele_file = "{}/{}".format(ele_files_dir, name.replace('-adv', '-ele'))
        clean_save_file(ele_file, ele_out_file)
        clean_save_file(adv_file, adv_out_file)

    dst = "muss/resources/datasets/osec_short_article"

    copyfile(ele_out_file, "{}/test.simple".format(dst))
    copyfile(adv_out_file, "{}/test.complex".format(dst))


def clean_save_file(src, dst):
    batch = []
    num_batch = 3
    with open(src, 'r') as f1, open(dst, 'a+') as f2:
        for line in f1:
            sentences = sent_tokenize(line)
            for sent in sentences:
                if num_batch <= 0:
                    output = "{}\n".format(" ".join(batch))
                    f2.write(output)
                    print(output)
                    return

                batch.append(sent.strip())
                num_batch = num_batch - 1


def count_lines_no_stop():
    file = "ts-discourse/out/cochrane/train.complex"
    count = 0
    with open(file, "r") as f:
        for line in f:
            if not re.match('.*\\.$', line):
                print(line)
                count = count + 1

    print(count)


def evaluate_coherence_complex():
    key = "_complex"
    evaluate_coherence(key)


def evaluate_coherence_outputs():
    key = ""
    evaluate_coherence(key)


def evaluate_coherence_ref():
    key = "_ref"
    evaluate_coherence(key)


def evaluate_coherence(key):
    coherence_main = config.get_coherence_cmd()
    coherence_model = config.get_coherence_model()
    model_type = "par_seq"
    glove_path = config.get_glove_path()
    token = "n/a"

    for name, model_path in models:
        output_dir = "{}/ts_discourse{}/{}".format(root, key, name)
        source_dir = "{}/ts_discourse/{}".format(root, name)
        print("Name: {} Model: {} Out: {}".format(name, model_path, output_dir))

        input_dir = output_dir
        if "complex" in key:
            input_dir = source_dir
        elif "ref" in key:
            input_dir = "{}/ts_discourse{}/{}".format(root, "", name)

        complex_files = glob.glob("{}/*.test.complex".format(input_dir))
        print(complex_files)
        for file in complex_files:
            cpx_name = os.path.basename(file)
            test_corpus = cpx_name.replace(".test.complex", "")
            coherence_dir = "{}/{}".format(output_dir, test_corpus)

            if not os.path.exists(coherence_dir):
                os.makedirs(coherence_dir)

            if "complex" in key:
                src = "{}/{}".format(source_dir, cpx_name)
                tgt = "{}/{}.csv".format(coherence_dir, cpx_name.replace(".complex", "").replace(".", "_"))
            elif "ref" in key:
                src = config.get_ref_file(test_corpus, "test")
                tgt = "{}/{}.csv".format(coherence_dir, cpx_name.replace(".complex", "").replace(".", "_"))
            else:
                pred_name = cpx_name.replace("complex", "pred")
                src = "{}/{}".format(output_dir, pred_name)
                tgt = "{}/{}.csv".format(coherence_dir, pred_name.replace(".pred", "").replace(".", "_"))

            print("Source: {} \nDest: {}\n".format(src, tgt))
            os.system("ln -sf {} {}".format(src, tgt))

            cmd = "python {} --data_dir {}/ --test_corpus {} --model_name {} --train_corpus {} --model_type {} " \
                  "--glove_path {} --task class --predict_only --predict_model {} --output_dir {}" \
                .format(coherence_main, output_dir, test_corpus, token, token, model_type, glove_path, coherence_model,
                        coherence_dir)
            print(cmd)
            os.system(cmd)


def evaluate_coherence_ref():
    coherence_main = config.get_coherence_cmd()
    coherence_model = config.get_coherence_model()
    model_type = "par_seq"
    glove_path = config.get_glove_path()
    token = "n/a"

    for name, model_path in models:
        print(name)
        output_dir = "{}/ts_discourse_complex/{}".format(root, name)
        source_dir = "{}/ts_discourse/{}".format(root, name)
        print(output_dir)

        complex_files = glob.glob("{}/*.test.complex".format(source_dir))
        print(complex_files)
        for file in complex_files:
            print()
            print(file)
            cpx_name = os.path.basename(file)
            pred_name = cpx_name.replace("complex", "pred")
            test_corpus = cpx_name.replace(".test.complex", "")
            coherence_dir = "{}/{}".format(output_dir, test_corpus)

            if not os.path.exists(coherence_dir):
                os.makedirs(coherence_dir)

            src = "{}/{}".format(source_dir, cpx_name)
            tgt = "{}/{}.csv".format(coherence_dir, cpx_name.replace(".complex", "").replace(".", "_"))
            print("Source: {} \nDest: {}\n".format(src, tgt))

            os.system("ln -sf {} {}".format(src, tgt))

            cmd = "python {} --data_dir {}/ --test_corpus {} --model_name {} --train_corpus {} --model_type {} --glove_path {} --task class --predict_only --predict_model {} --output_dir {}" \
                .format(coherence_main, output_dir, test_corpus, token, token, model_type, glove_path, coherence_model,
                        coherence_dir)
            print(cmd)

            os.system(cmd)


def summarize_coherence(dirname):
    result = []
    for model_name, _ in models:
        output_dir = "{}/{}/{}".format(root, dirname, model_name)
        complex_files = glob.glob("{}/*/*.csv".format(output_dir))
        print(complex_files)
        for cpx_name in complex_files:
            test_corpus = Path(cpx_name.replace("_test.csv", "")).name.strip()
            coherence_file = "{0}/{3}/{1}/{2}/coherence_eval_{1}_{2}.txt".format(root, model_name, test_corpus, dirname)
            print("Coherence File: {}".format(coherence_file))

            if Path(coherence_file).exists():
                result.append("{},{},{}".format(model_name, test_corpus, get_coherence_score(coherence_file)))
            else:
                print("File does not exist: {}".format(coherence_file))

    with open("{}/coherence_report_{}.csv".format(root, dirname), "w", encoding="utf-8") as f:
        for line in result:
            f.write("{}\n".format(line))


def show_coherence_distribution(dirname, key):
    # result = []
    for model_name, _ in models:
        output_dir = "{}/{}/{}".format(root, dirname, model_name)
        complex_files = glob.glob("{}/{}/*.csv".format(output_dir, key))
        # complex_files = [c for c in complex_files if not ("asset" in c or "sentence" in c)]
        complex_files = [c for c in complex_files if key in c]
        print(complex_files)
        for cpx_name in complex_files:
            test_corpus = Path(cpx_name.replace("_test.csv", "")).name.strip()
            coherence_file = "{0}/{3}/{1}/{2}/coherence_eval_{1}_{2}.txt".format(root, model_name, test_corpus, dirname)
            print("Coherence File: {}".format(coherence_file))

            if Path(coherence_file).exists():
                # result.append("{},{},{}".format(model_name, test_corpus, get_coherence_values(coherence_file)))
                scores = get_coherence_values(coherence_file)
                sns.distplot(scores, hist=False, kde=True,
                             kde_kws={'linewidth': 2},
                             label=model_alias[model_name])
                # color=colors[i])
            else:
                print("File does not exist: {}".format(coherence_file))

    plt.xticks([-1, 0, 1], ["Low", "Medium", "High"])
    plt.title("Test Set: {}".format(model_alias[key]))
    plt.show()
    plt.clf()


data_rename = {"en_wikilarge_mined": "Mined+WikiLarge",
               "en_wikilarge_cochrane_best": "Mined+WikiLarge+Cochrane(best)",
               "en_wikilarge_cochrane_last": "Mined+WikiLarge+Cochrane(last)",
               "en_mined_cochrane_best": "Mined+Cochrane(best)",
               "en_mined_cochrane_last": "Mined+Cochrane(last)",
               "en_wikilarge_mined_d_wiki_best": "Mined+WL+dWiki(best)",
               "en_wikilarge_mined_d_wiki_last": "Mined+WL+dWiki(last)",
               "en_mined_d_wiki_last": "Mined+dWiki(last)",
               "en_mined_d_wiki_best": "Mined+dWiki(best)",
               "en_mined_d_wiki_last_2": "Mined+dWiki2(last)",
               "en_mined_d_wiki_best_2": "Mined+dWiki2(best)",
               "en_mined_cochrane_d_wiki_last": "Mined+Cochrane+dWiki(last)",
               "en_mined_cochrane_d_wiki_best": "Mined+Cochrane+dWiki(best)",
               "osec_short_article": "OneStop_Article",
               "osec_sentence_aligned": "OneStop_Sentence",
               "en_mined": "Mined",
               "asset": "Asset",
               "cochrane": "Cochrane",
               "d_wikipedia": "D-Wiki"
               }


def get_sari_bleu_fkgl_dsari_report():
    sent_metrics_file = "{}/ts_discourse/ts_discourse.csv".format(root)
    dsari_metric_file = "{}/ts_discourse/d_sari.csv".format(root)

    df_sent = pd.read_csv(sent_metrics_file, header=0, names=["model", "test", "bleu", "sari", "fkgl"])
    df_dsari = pd.read_csv(dsari_metric_file, header=0, names=["model", "test", "dsari", "dkeep", "ddelete", "dadd"])

    report = pd.merge(df_sent, df_dsari, how='inner', on=["model", "test"])
    report["dsari"] = report["dsari"].apply(lambda x: x * 100)
    report["test"] = report["test"].apply(lambda x: x.replace(".test.complex", ""))
    report[["bleu", "fkgl", "sari", "dsari", "dkeep", "ddelete", "dadd"]] = \
        report[["bleu", "fkgl", "sari", "dsari", "dkeep", "ddelete", "dadd"]].apply(lambda x: round(x, 3))

    report["alias"] = report["model"].apply(lambda x: data_rename[x])

    final = report[["model", "alias", "test", "bleu", "fkgl", "sari", "dsari", "dkeep", "ddelete", "dadd"]]
    final.to_csv("{}/ts_discourse/sari_bleu_fkgl_dsari_report.csv".format(root), index=False, header=True)
    print(final)


def coherence_complex_simple_ref_report():
    complex_file = "{}/coherence_report_ts_discourse_complex.csv".format(root)
    simple_file = "{}/coherence_report_ts_discourse.csv".format(root)
    ref_file = "{}/coherence_report_ts_discourse_ref.csv".format(root)

    df_complex = pd.read_csv(complex_file, names=["model", "test", "c_score"])
    df_simple = pd.read_csv(simple_file, names=["model", "test", "s_score"])
    df_ref = pd.read_csv(ref_file, names=["model", "test", "r_score"])

    report = pd.merge(df_complex, df_simple, how='inner', on=["model", "test"])
    report = pd.merge(report, df_ref, how='inner', on=["model", "test"])

    report[["c_score", "s_score", "r_score"]] = report[["c_score", "s_score", "r_score"]].apply(lambda x: round(x, 3))
    report["alias"] = report["model"].apply(lambda x: data_rename[x])

    # report = report.sort_values(by="diff", ascending=False)
    report.to_csv("{}/coherence_complex_simple_ref_report.csv".format(root), index=False, header=True)
    print(report)


def sari_fkgl_all_report():
    sari_file = "d_sari2.csv"
    fkgl_file = "ts_fkgl_all.csv"

    df_sari = pd.read_csv(sari_file)  # , names=["model", "dataset", "sari", "keep", "delete", "add"])
    df_fkgl = pd.read_csv(fkgl_file)  # , names=["model", "dataset", "fkgl_c", "fkgl_s", "fkgl_ref"])
    df_sari["dataset"] = df_sari["dataset"].apply(lambda c: c.replace(".test.complex", ""))
    report = pd.merge(df_sari, df_fkgl, how='inner', on=["model", "dataset"])

    # report["diff"] = report["score_x"] - report["score_y"]
    # report["score_y"] = report["score_y"].apply(lambda x: round(x, 3)

    for col in ["sari", "keep", "delete", "add", "fkgl_c", "fkgl_s", "fkgl_ref"]:

        if "fkgl" in col:
            report[col] = report[col].apply(lambda x: round(x, 2))
        else:
            report[col] = report[col].apply(lambda x: round(x * 100, 2))

    report = report.sort_values(by=["dataset", "model"], ascending=False)

    report = report[(report["model"] == "en_d_wiki_v2_last") |
                    (report["model"] == "en_dwiki_cochrane_last") |
                    (report["model"] == "en_mined") |
                    (report["model"] == "en_cochrane_last")]
    report = report.to_csv("table_2_sari_fkgl", index=False)
    print(report)


def visualize_coherence():
    report = "{}/coherence_report.csv".format(root)
    df = pd.read_csv(report, names=["Model", "Test Set", "Coherence"])
    df_by_score = df.sort_values(by="Coherence", ascending=False)
    df_by_test = df_by_score
    # df_by_test = df_by_score.sort_values(by="Test Set", ascending=False)
    df_by_test['Model'] = df_by_test['Model'].apply(lambda f: f.replace(f, data_rename[f]))
    df_by_test['Test Set'] = df_by_test['Test Set'].apply(lambda f: f.replace(f, data_rename[f]))
    df_by_test['Coherence'] = df_by_test['Coherence'].apply(lambda f: round(f, 3))

    df_by_test["latex"] = df_by_test["Model"] + " & " + df_by_test["Test Set"] + " & " \
                          + df_by_test['Coherence'].astype(str) + " \\\\ \n\hline"

    result = "\n".join(df_by_test["latex"].values).replace("_", "-")
    print(result)

    print(df_by_test)


def visualize_all_coherence(latex=False):
    report = "{}/out/coherence_complex_simple_ref_report.csv".format(root)
    output_dir = os.path.dirname(report)

    df = pd.read_csv(report, header="infer")
    df = df.sort_values(by=["c_score", "s_score", "r_score"], ascending=False)
    df["model"] = df["model"].apply(lambda f: f.replace(f, data_rename[f]))
    df["test"] = df["test"].apply(lambda f: f.replace(f, data_rename[f]))
    df[["c_score", "s_score", "r_score"]] = df[["c_score", "s_score", "r_score"]].apply(lambda f: round(f, 2))
    df = df.sort_values(by=["test", "c_score", "s_score", "r_score"], ascending=False)
    df = df.drop(columns=["alias"])

    if latex:
        latex_out = "\n\\hline\n" + " & ".join(["Model", "Test Set", "Complex", "Simple", "Reference"]) + \
                    "\\\\\n\\hline \\hline\n"
        for index, line in df.iterrows():
            latex_out = latex_out + " & ".join([str(s) for s in line.values]) + " \\\\ \n\\hline\n"
        print(latex_out)
        with open("{}/ts_coherence.latex".format(output_dir), "w") as f1:
            f1.write(latex_out)


def get_normalized_coherence(value):
    # low = -1, med = 0, high = +1
    if value == 2:
        return 1
    elif value == 1:
        return 0
    elif value == 0:
        return -1
    else:
        raise "Invalid Coherence score: {}".format(value)


def get_coherence_score(file):
    df = pd.read_csv(file, names=["Sentence", "Coherence"], sep="\t")
    df["Scores"] = df["Coherence"].apply(lambda f: get_normalized_coherence(f))
    score = df["Scores"].mean()

    return score


def get_coherence_values(file):
    df = pd.read_csv(file, names=["Sentence", "Coherence"], sep="\t")
    df["Scores"] = df["Coherence"].apply(lambda f: get_normalized_coherence(f))
    scores = df["Scores"].values

    return scores


def collect_manual_outputs():
    root = config.get_muss_root()

    models = []

    for model in ["en_mined_cochrane_last", "en_mined_d_wiki_last"]:
        for test_set in ["cochrane", "osec_short_article", "d_wikipedia"]:
            models.append("{0}/ts_discourse/{1}/{2}/coherence_eval_{1}_{2}.txt".format(root, model, test_set))

    output = []
    extract_sample(models, 2, "high", output)
    extract_sample(models, 0, "low", output)
    pd.concat(output).to_csv("manual_outputs_analysis.csv", index=False, sep="\t")


def extract_sample(files, coherence_value, coherence_label, output):
    for file in files:
        name = Path(file).name
        df = pd.read_csv(file, sep='\t', names=['Sentence', 'Coherence'])
        df_high = df.loc[df['Coherence'] == coherence_value]
        result = df_high.sample(n=25, random_state=100)
        result["Source"] = name
        output.append(result)
        # result["Sentence"].to_csv("{}/{}.{}.manual.csv".format(root, name, coherence_label),
        #                           index=False, header=False)
    return output


def get_fkgl(complex_file, system_file, reference_file):
    local_results = {}
    for sample, label in zip([complex_file, system_file, reference_file], ["complex", "system", "ref"]):
        score_all = 0
        count_all = 0
        with open(sample) as f1:
            for line in f1:
                score_all += textstat.flesch_kincaid_grade(line)
                count_all += 1
        score_all = float(score_all / count_all)
        local_results[label] = round(score_all, 6)

    return local_results


def run_d_sari():
    script = "python Document-level-text-simplification/D_SARI.py"

    root_local = "output/paper_v2"
    output_dir = "ts_discourse/"
    easse_report = "{}/dsari2_report.out".format(root_local)

    if os.path.exists(easse_report):
        os.remove(easse_report)

    complex_files = glob.glob("{}/*/*test.complex".format(output_dir))
    complex_files = list(filter(lambda c: "short_article" not in c, complex_files))
    complex_files = list(filter(lambda c: "best" not in c, complex_files))
    complex_files = list(filter(lambda c: "wikilarge" not in c, complex_files))
    complex_files = list(filter(lambda c: "en_d_wiki_last" not in c, complex_files))
    # complex_files = list(filter(lambda c: "osec_all" in c, complex_files))

    for file in complex_files:
        complex, system, reference, dataset, model = get_sets(file)
        cmd = "{} --complex {} --simple {} --ref {}".format(
            script, complex, system, reference)
        print(cmd)
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
        print(result.stderr)

        fkgl_scores = get_fkgl(complex, system, reference)
        with open(easse_report, 'a+') as f:
            out = result.stdout.decode("utf-8")
            out = out.replace("[", "").replace("]", "").replace(",", "")
            sari, keep, delete, add = out.strip().split()
            f.write("{}\n".format(",".join([model, dataset, sari, keep, delete, add,
                                            str(fkgl_scores["complex"]), str(fkgl_scores["system"]),
                                            str(fkgl_scores["ref"])])))


def format_report_to_latex():
    with open("dsari2_report.out") as f1:
        for line in f1:
            model, test_set, s1, s2, s3, s4, f1, f2, f3 = line.strip().split(",")
            s1 = round(float(s1) * 100, 2)
            s2 = round(float(s2) * 100, 2)
            s3 = round(float(s3) * 100, 2)
            s4 = round(float(s4) * 100, 2)
            f1 = round(float(f1), 2)
            f2 = round(float(f2), 2)
            f3 = round(float(f3), 2)

            line = f"{model} & {test_set} & {s1} & {s2} & {s3} & {s4} & {f1} & {f2} & {f3}"
            print(f"{line}\\\\ \n\\hline")
            # en_cochrane_last, osec_all, 0.2605, 0.148785, 0.627406, 0.005308, 10.705556, 9.837037, 7.892593


def get_sets(complex_file):
    path = "ts_coherence"
    path_ref = "ts_coherence/datasets"
    root_name = os.path.basename(complex_file).replace("complex", "pred")
    model = Path(complex_file).parent.name
    dataset = Path(complex_file).name.split(".")[0]
    reference = f"{path_ref}/{dataset}/test.simple"
    system = complex_file.replace("complex", "pred")

    return complex_file, system, reference, dataset, model


if __name__ == '__main__':
    # Step 1: Train the model
    # time python scripts/train_model.py (From MUSS repo)

    # Step 2: Evaluate the models
    # setup_dirs()

    # Step 3: Run EASSE on system outputs
    # run_easse()

    # Step 4: Print Latex table and *png files with results
    # show_results()
    # show_results(latex=True)

    # Step 5: Evaluate coherence system outputs
    # Run d-wiki models coherence outputs - 3431139
    # evaluate_coherence_outputs()
    # evaluate_coherence_complex()
    # evaluate_coherence_ref() # running

    # Step 5.1: Collect coherence outputs
    # summarize_coherence("ts_discourse")
    # summarize_coherence("ts_discourse_complex")
    # coherence_complex_simple_report()
    # summarize_coherence()
    # visualize_coherence()
    # collect_manual_outputs()

    # show_coherence_distribution("ts_discourse", "osec_short_article")
    # show_coherence_distribution("ts_discourse", "cochrane")
    # show_coherence_distribution("ts_discourse", "d_wikipedia")
    # show_coherence_distribution("ts_discourse", "asset")
    # show_coherence_distribution("ts_discourse", "osec_sentence_aligned")
    # summarize_coherence("ts_discourse_ref")

    # Step 6: unify reports: complex, simple, reference.
    coherence_complex_simple_ref_report()

    # Step 7: Run D-Sari Metric
    # run_d_sari()

    # Step 8: Collect d-sari and sentence-based metrics
    # get_sari_bleu_fkgl_dsari_report()

    # Step 9: Collect manual samples
    # collect_manual_outputs()
