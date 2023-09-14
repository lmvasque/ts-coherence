# Document-level Text Simplification with Coherence Evaluation

Source code for the
paper: [Document-level Text Simplification with Coherence Evaluation](https://tsar-workshop.github.io/program/papers/vasquez-rodriguez-etal-2023-document.pdf)
accepted in the Second Workshop on Text Simplification, Accessibility and
Readability [(TSAR 2023)](https://tsar-workshop.github.io/)
at the Recent Advances in Natural Language Processing Conference [(RANLP 2023)](http://ranlp.org/ranlp2023/).

By [@lmvasquezr](https://twitter.com/lmvasquezr), [@MattShardlow](https://twitter.com/MattShardlow)
, [Piotr Przybyła](https://home.ipipan.waw.pl/p.przybyla/) and [@SAnaniadou](https://twitter.com/SAnaniadou). If you
have any questions, please don't hesitate to [contact us](mailto:lvasquezcr@gmail.com).

## Setup

This code was tested using **Python 3.7+**. You can setup our repo as follows:

```bash
git clone https://github.com/lmvasque/ts-coherence.git
cd ts-coherence
pip install -r requirements.txt
```

## Datasets

We have selected the `D-Wikipedia` and `Cochrane` datasets for training. For testing, we have used the `OneStopCorpus`.
These datasets can be downloaded from:

- D-Wikipedia: https://github.com/rlsnlp/document-level-text-simplification
- Cochrane: https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts
- OneStopCorpus: https://github.com/nishkalavallabhi/OneStopEnglishCorpus


## Models
### 1. Fine-tuning of MUSS model

We fine-tuned the [MUSS model](https://github.com/lmvasque/muss) by using the
[script](https://github.com/facebookresearch/muss/blob/main/scripts/train_model.py) below. You can refer to the setup of
the model in the original MUSS model repo.

```
python scripts/train_model.py
```

We have run with the original [code](https://github.com/lmvasque/muss), including minor modifications in
[train_model.py](https://github.com/lmvasque/muss/blob/main/scripts/train_model.py) to extend the length of the outputs.
This setting is already set in the latest MUSS model. Also, we have modified the file [training.py](https://github.com/facebookresearch/muss/blob/main/muss/mining/training.py) to add our datasets for training and test.

### 2. Generation of system outputs

After the MUSS model is fine-tuned, we generate our simplifications based on the code from this [script](https://github.com/facebookresearch/muss/blob/main/scripts/simplify.py):
```
python scripts/simplify.py
```

This code is limited to work with the original models, so additional changes are needed to add new fine-tuned models.

### 3. Simplifications evaluation

For the **FKGL** scores, we evaluated the model outputs using [EASSE](https://github.com/feralvam/easse):

```
easse evaluate --orig_sents_path complex.txt --test_set custom -i simplified.txt --refs_sents_paths references.txt
```

For **D-SARI**, we used the evaluation scripts published on
the [original repository](https://github.com/rlsnlp/document-level-text-simplification).

For the **Coherence** evaluation, we requested the code & dataset for the GCDC
corpus [here](https://github.com/aylai/GCDC-corpus).

From this code, we retrained the Parseq model using the Yahoo dataset as follows:

```
python main.py --model_name yahoo_class_model --train_corpus Yahoo --model_type par_seq --task class
```

### 4. About the source code

We include our source code as a reference to the developed solution for our specific case. The code is mostly related to the evaluation and analysis of our results. 
Nevertheless, we consider that the overall idea can be applied straight forward to any research following the steps above. 
We are happy to answer any question :)


## Citation

If you use our results in your research, please cite our work: [Document-level Text Simplification with Coherence Evaluation](https://tsar-workshop.github.io/program/papers/vasquez-rodriguez-etal-2023-document.pdf) 


```
@inproceedings{vasquez-rodriguez-etal-2023-document,
    title = "Document-level Text Simplification with Coherence Evaluation",
    author = "V{\'a}squez-Rodr{\'\i}guez, Laura  and
      Shardlow, Matthew  and
      Przyby{\l}a, Piotr  and
      Ananiadou, Sophia",
    booktitle = "Proceedings of the Second Workshop on Text Simplification, Accessibility, and Readability (TSAR-2023)",
    month = sept,
    year = "2023",
    address = "Varna, Bulgaria",
    url = "https://tsar-workshop.github.io/program/papers/vasquez-rodriguez-etal-2023-document.pdf"
}
