import glob
import re
from pathlib import Path

import pandas as pd
from pyirr import kappam_fleiss

MODELS = {'cochrane_last_d_wikipedia': 'A', 'dwiki_cochrane_last_d_wikipedia': 'B', 'dwiki_cochrane_last_osec_all': 'C',
          'mined_osec_all': 'D', 'cochrane_last_cochrane': 'E', 'd_wiki_v2_last_d_wikipedia': 'F',
          'mined_cochrane': 'G', 'dwiki_cochrane_last_cochrane': 'H', 'mined_d_wikipedia': 'I',
          'd_wiki_v2_last_osec_all': 'J', 'cochrane_last_osec_all': 'K', 'd_wiki_v2_last_cochrane': 'L'}

MODELS_FOR_EVAL = {'d_wiki_v2_last_d_wikipedia': 'F', 'cochrane_last_cochrane': 'E'}

TASK_TEXT = "Please rank the texts below according to the following scale of coherence: low or high"

EVAL_TEXT = "What is a coherent text? -- A text with high coherence is easy to understand, well-organized, and contains only details that " \
            "support the main point of the text. A text with low coherence is difficult to understand, not well " \
            "organized, or contains unnecessary details. Try to ignore the effects of grammar or spelling errors " \
            "when assigning a coherence rating."


def create_human_evaluation_corpus():
    PATH = "output/human_evaluation"

    files = glob.glob(f"{PATH}/*/*txt")

    headers = ["index", "model", "type", "score", "text"]
    df = pd.DataFrame(columns=headers)
    for file in files:
        df_csv = pd.read_csv(file, sep="\t", names=["text", "score"])
        name = Path(file).name
        simp_type = Path(file).parent.name
        name = name.replace("coherence_eval_en_", "")
        name = name.replace(".txt", "")
        df_csv["model"] = name
        df_csv["type"] = simp_type
        df_csv["index"] = range(0, len(df_csv))
        df = pd.concat([df, df_csv[headers]])

    df.to_csv(f"{PATH}/coherence_to_review.csv", index=False, sep="\t")


def get_models_naming(models_list):
    models_naming = {}
    code = 65
    for model in models_list:
        models_naming[model] = chr(code)
        code += 1

    print(models_naming)


def get_normalised_autoscore(label):
    label = int(label)
    if label == 0:
        return -1
    if label == 1:
        return 0
    elif label == 2:
        return 1

    return 0


def get_human_evaluation_autoscores():
    PATH = "output/human_evaluation/human_evaluation_raw_ipm.xlsx"
    df = pd.read_excel(PATH, names=["sample_id", "model", "type", "auto_score", "text"])
    df["model_code"] = df["model"].apply(lambda f: MODELS[f])
    df["id"] = df["sample_id"].astype("string") + "_" + df["model_code"].astype("string")
    print(df.head())
    df = df[(df["model_code"] == "E") | (df["model_code"] == "F")]
    df["score_n"] = df["auto_score"].apply(lambda f: get_normalised_autoscore(f))
    df = df.drop(columns=["model", "sample_id", "model_code", "type", "auto_score"])
    df.to_csv(f"{PATH}.autoscores.csv", index=False)


def prepare_file_for_eval():
    num_annotators = 15
    num_samples = 50
    num_systems = 2
    num_unique_files = 5
    PATH = "output/human_evaluation/human_evaluation_raw_ipm.xlsx"
    df = pd.read_excel(PATH, names=["sample_id", "model", "type", "auto_score", "text"])
    df = df.drop(columns=["type", "auto_score"])
    df["model_code"] = df["model"].apply(lambda f: MODELS[f])
    df["id"] = df["sample_id"].astype("string") + "_" + df["model_code"].astype("string")
    df["coherence_rank"] = ""
    print(df.head())

    human_files = []
    for user in range(0, num_unique_files):
        empty_df = pd.DataFrame()
        human_files.append(empty_df)

    models = sorted(list(set(df["model"].values)))
    for model in models:
        if model in MODELS_FOR_EVAL.keys():
            df_model = df[df["model"] == model]
            df_sample = df_model.sample(n=num_unique_files * 20, random_state=324)
            count = 0
            for i in range(0, 10):
                for j in range(0, num_unique_files):
                    human_files[j] = human_files[j].append(df_sample.iloc[count])
                    count = count + 1

    for i, human in enumerate(human_files):
        df = human[["id", "text"]]
        out = f"{Path(PATH).parent}/coherence_evaluation_{i}.xlsx"
        to_formatted_excel(df, out)


def to_formatted_excel(df, path):
    # df = pd.read_csv(path)

    df = df.sort_values(by=['id'], ascending=False, key=lambda col: col.str[-1])
    df_task = df.copy()[0:4]
    df_task["id"] = ""
    df_task["text"] = ""

    df = df_task.append(df)
    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Evaluation', index=None)
    workbook = writer.book
    worksheet = writer.sheets['Evaluation']

    header_format = workbook.add_format({'bold': True,
                                         'font_size': 18,
                                         'border': 1,
                                         'center_across': True,
                                         'bg_color': "#9cbc5c",
                                         'font_color': "#f6f9ed"
                                         })

    text_format = workbook.add_format({'text_wrap': True,
                                       'font_size': 14,
                                       'shrink': True})

    text_format2 = workbook.add_format({'text_wrap': True,
                                        'font_size': 14,
                                        'shrink': True,
                                        'center_across': True,
                                        'bold': False})

    task_format = workbook.add_format({'text_wrap': True,
                                       'font_size': 16,
                                       'shrink': True,
                                       'border': 1,
                                       'bg_color': "#ededed"})

    worksheet.add_table('A5:C25', {'autofilter': False,
                                   'first_column': True,
                                   'style': 'Table Style Medium 4',
                                   'columns': [{'header': '#',
                                                'header_format': header_format,
                                                'format': text_format2},
                                               {'header': 'Text',
                                                'header_format': header_format,
                                                'format': text_format},
                                               {'header': 'Coherence Rank',
                                                'header_format': header_format,
                                                'format': text_format2}]})

    worksheet.set_column("A:A", 0, text_format2)
    # worksheet.set_column("A:A", None, {'hidden': True})
    worksheet.set_column("B:B", 150, text_format)
    worksheet.set_column("C:C", 30, text_format2)

    # Merge task rows
    worksheet.merge_range('A1:C1', "Coherence-based Human Evaluation", header_format)
    worksheet.merge_range('A2:C2', TASK_TEXT, task_format)
    worksheet.merge_range('A3:C3', EVAL_TEXT, task_format)
    worksheet.merge_range('A4:C4', " ", None)

    # Set width of task rows
    worksheet.set_row(0, None, None)
    worksheet.set_row(1, 20, None)
    worksheet.set_row(2, 70, None)

    combo_code = {'validate': 'list', 'source': ['low', 'high']}
    worksheet.freeze_panes(1, 0)
    worksheet.freeze_panes(2, 0)
    worksheet.freeze_panes(3, 0)
    worksheet.freeze_panes(4, 0)
    worksheet.freeze_panes(5, 0)

    for i in range(6, 21):
        worksheet.data_validation(f"C{i}", combo_code)

    writer.save()


def read_excel(path):
    df = pd.read_excel(f"{path}/master_file.xlsx")
    df.head()


def get_score(label):
    if "low" in label:
        return -1
    elif "high" in label:
        return 1

    return 0


def calculate_auto_vs_human(evaluations):
    auto_scores = "output/human_evaluation/human_evaluation_raw_ipm.xlsx.autoscores.csv"
    df_auto = pd.read_csv(auto_scores)

    df = evaluations[["id", "score_n"]].merge(df_auto[["id", "score_n"]], how='inner', on='id')
    df.columns = ["id", "human", "auto"]
    df = df.astype({"human": float, "auto": float})

    df_general = df[df["id"].str.contains("_F")]
    df_general = df_general.drop(columns=["id"])
    df_general.to_csv(
        "output/human_evaluation/human_auto_scores_general.csv",
        index=False)

    df_medical = df[df["id"].str.contains("_E")]
    df_medical = df_medical.drop(columns=["id"])
    df_medical.to_csv(
        "output/human_evaluation/human_auto_scores_medical.csv",
        index=False)

    return df


def process_all_evaluations():
    path = "human_evaluation/"
    files = glob.glob(f"{path}/*xlsx*")

    evaluations = pd.DataFrame()
    annotations = {}
    # {1: "", 2: "", 3: "", 4: ""}

    for i, file in enumerate(files):
        eval_num = int(re.match(r".*coherence_evaluation_(\d+).*", file).group(1))
        df = pd.read_excel(file)
        df = df.drop([0, 1, 2, 3])
        evaluations = pd.concat([evaluations, df])
        scores = list(df.iloc[:, 2])

        if eval_num not in annotations.keys():
            annotations[eval_num] = pd.DataFrame()

        annotations[eval_num][f"rater{i}"] = [get_score(f) for f in scores]

    calculate_iaa(annotations)
    evaluations.columns = ["id", "text", "score_c", "comments"]
    evaluations["model"] = evaluations["id"].apply(lambda f: f.split("_")[1])
    evaluations["score_n"] = evaluations["score_c"].apply(lambda f: get_score(f))
    evaluations = evaluations.drop(columns=["comments", "text"])
    evaluations = evaluations.groupby(["id", "model"])["score_n"].mean().reset_index()

    calculate_auto_vs_human(evaluations)

    eval_by_model = evaluations.groupby(["model"])["score_n"].mean().reset_index()
    print(eval_by_model)

    return evaluations


def calculate_iaa(annotations):
    all_annotations_general = pd.DataFrame()
    all_annotations_medical = pd.DataFrame()
    for set_i, df in annotations.items():
        df.columns = ["rater1", "rater2", "rater3"]
        all_annotations_general = pd.concat([all_annotations_general, df[0:10]])
        all_annotations_medical = pd.concat([all_annotations_medical, df[10:len(df)]])

    for d, label in zip([all_annotations_general, all_annotations_medical], ["general", "medical"]):
        out = kappam_fleiss(d, detail=True)
        kappa = out.value
        z = out.statistic
        p = out.pvalue

        with open("../output/human_evaluation/iaa.csv", "a+") as f1:
            f1.write(f"{round(kappa, 3)},{round(z, 3)},{round(p, 3)},{label}\n")


process_all_evaluations()
