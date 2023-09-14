import json


class Config:

    def __init__(self):
        self.path = "config/models/muss_models.json"
        self.data = self.load()
        pass

    def load(self):
        with open(self.path) as f:
            data = json.load(f)
            f.close()
        return data

    def get_models(self):
        models = []
        base_name = self.data['global']['models_path']
        print("Models_Path: ".format(base_name))
        for model in self.data['models']:
            path = self.data['models'][model]
            models.append((model, "{}/{}".format(base_name, path)))
            print(path)
        return models

    def get_complex_sentences(self):
        data = self.data['global']['data_path']
        complex_sent = []

        for test_set in self.data['data']:
            test_file = "{}/{}/{}".format(data, test_set, self.data['data'][test_set]['test'][0])
            complex_sent.append(test_file)

            if "valid" in self.data['data'][test_set]:
                valid_file = "{}/{}/{}".format(data, test_set, self.data['data'][test_set]['valid'][0])
                complex_sent.append(valid_file)

        return ",".join(complex_sent)

    def get_ref_file(self, dataset, subset):
        data_path = self.data['global']['data_path']
        file = self.data['data'][dataset][subset][1]
        ref_file = "{}/{}/{}".format(data_path, dataset, file)

        return ref_file

    def get_muss_root(self):
        return self.data['global']['root']

    def get_easse_cmd(self):
        return self.data['global']['easse_bin']

    def get_dsari_cmd(self):
        return self.data['global']['d_sari_bin']

    def get_eval_sets(self):
        return list(self.data['data'].keys())

    def get_model_alias(self):
        return self.data['alias']

    def get_coherence_cmd(self):
        return self.data['global']['coherence_bin']

    def get_coherence_model(self):
        return self.data['global']['coherence_model']

    def get_glove_path(self):
        return self.data['global']['glove_path']
