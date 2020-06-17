import json
import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from stud.utilities import load_pickle, save_pickle


class JSONDataParser(Dataset):
    def __init__(self, file_name, device):
        self.encoded_data = []
        self.data_x, self.data_y = [], []
        self.pos_x, self.predicates_x = [], []
        self.device = device
        self.sentences, self.labels = self.read_data(file_name)

    def read_data(self, path):
        with open(path) as f:
            dataset = json.load(f)

        sentences, labels = {}, {}
        for sentence_id, sentence in tqdm(dataset.items(), desc='Parsing dataset', leave=False):
            sentence_id = int(sentence_id)
            sentences[sentence_id] = {
                'words': sentence['words'],
                'lemmas': sentence['lemmas'],
                'pos_tags': sentence['pos_tags'],
                'dependency_heads': [int(head) for head in sentence['dependency_heads']],
                'dependency_relations': sentence['dependency_relations'],
                'predicates': sentence['predicates'],
            }

            labels[sentence_id] = {
                'predicates': sentence['predicates'],
                'roles': {int(p): r for p, r in sentence['roles'].items()}
            }
        return sentences, labels

    def get_sentences(self):
        for i in tqdm(range(len(self.sentences)), desc='Fetching sentences and labels', leave=False):
            roles_ = self.labels[i].get('roles')
            for _, k in enumerate(roles_):
                lemmas_sentence_ = self.sentences[i].get('lemmas')[:]
                predicates_sentence = self.sentences[i].get('predicates')[:]
                pos_sentence = self.sentences[i].get('pos_tags')[:]

                verb_atlas_grp = predicates_sentence[k]
                predicates_sentence_ = ["_"] * len(predicates_sentence)
                predicates_sentence_[k] = verb_atlas_grp

                self.data_x.append(lemmas_sentence_)
                self.predicates_x.append(predicates_sentence_)
                self.pos_x.append(pos_sentence)
                self.data_y.append(roles_.get(k))

    @staticmethod
    def build_vocabulary(data_x, load_from=None):
        if load_from and Path(load_from).is_file():
            stoi = load_pickle(load_from)
            itos = {key: val for key, val in enumerate(stoi)}
            return stoi, itos
        all_words = [item for sublist in data_x for item in sublist]
        unigrams = sorted(list(set(all_words)))
        stoi = {'<PAD>': 0, '<UNK>': 1}
        start_ = 2
        stoi.update({val: key for key, val in enumerate(unigrams, start=start_)})
        itos = {key: val for key, val in enumerate(stoi)}
        save_pickle(load_from, stoi)
        save_pickle(load_from.replace('stoi', 'itos'), itos)
        return stoi, itos

    @staticmethod
    def build_labels_vocabulary(data_y, load_from=None):
        if load_from and Path(load_from).is_file():
            label2idx = load_pickle(load_from)
            idx2label = {key: val for key, val in enumerate(label2idx)}
            return label2idx, idx2label
        all_words = [item for sublist in data_y for item in sublist]
        unigrams = sorted(list(set(all_words)))
        label2idx = {'<PAD>': 0}
        start_ = 1
        label2idx.update({val: key for key, val in enumerate(unigrams, start=start_)})
        idx2label = {key: val for key, val in enumerate(label2idx)}

        save_pickle(load_from, label2idx)
        save_pickle(load_from.replace('label2idx', 'idx2label'), idx2label)
        return label2idx, idx2label

    def encode_dataset(self, word2idx, label2idx, pos2idx, predicate2idx):
        data_x_stoi, data_y_stoi = [], []
        pos_x_stoi, predicates_x_stoi = [], []
        for sentence, labels, sentence_pos, sentence_predicate in tqdm(zip(self.data_x, self.data_y, self.pos_x, self.predicates_x),
                                                                       desc='Numericalizing Data', leave=False, total=len(self.data_x)):
            data_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in sentence]).to(self.device))
            data_y_stoi.append(torch.LongTensor([label2idx.get(tag) for tag in labels]).to(self.device))
            pos_x_stoi.append(torch.LongTensor([pos2idx.get(pos_tag, 1) for pos_tag in sentence_pos]).to(self.device))
            predicates_x_stoi.append(torch.LongTensor([predicate2idx.get(predicate, 1) for predicate in sentence_predicate]).to(self.device))

        for i in range(len(data_x_stoi)):
            self.encoded_data.append({'inputs': data_x_stoi[i],
                                      'pos': pos_x_stoi[i],
                                      'predicates': predicates_x_stoi[i],
                                      'outputs': data_y_stoi[i]})

    def get_element(self, idx):
        return self.data_x[idx], self.data_y[idx]

    @staticmethod
    def pad_batch(batch):
        inputs_batch = [sample["inputs"] for sample in batch]
        predicates_inputs_batch = [sample["predicates"] for sample in batch]
        pos_inputs_batch = [sample["pos"] for sample in batch]
        outputs_batch = [sample["outputs"] for sample in batch]

        return {'inputs': pad_sequence(inputs_batch, batch_first=True),
                'pos': pad_sequence(pos_inputs_batch, batch_first=True),
                'predicates': pad_sequence(predicates_inputs_batch, batch_first=True),
                'outputs': pad_sequence(outputs_batch, batch_first=True)}

    @staticmethod
    def decode_predictions(predictions, idx2label):
        predictions_ = predictions.tolist()
        prediction_ = []
        for prediction in predictions_:
            prediction_.append([idx2label.get(tag) for tag in prediction])
        return prediction_

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data', 'train.json')
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_set = JSONDataParser(path, device=_device)
    training_set.get_sentences()

    word2idx_path = os.path.join(os.getcwd(), 'model', 'word_stoi.pkl')
    word2idx, idx2word = JSONDataParser.build_vocabulary(training_set.data_x, word2idx_path)

    pos2idx_path = os.path.join(os.getcwd(), 'model', 'pos_stoi.pkl')
    pos2idx, idx2pos = JSONDataParser.build_vocabulary(training_set.pos_x, pos2idx_path)

    predicate2idx_path = os.path.join(os.getcwd(), 'model', 'predicate_stoi.pkl')
    predicate2idx, idx2predicate = JSONDataParser.build_vocabulary(training_set.predicates_x, predicate2idx_path)

    label2idx_path = os.path.join(os.getcwd(), 'model', 'label2idx.pkl')
    label2idx, idx2label = JSONDataParser.build_labels_vocabulary(training_set.data_y, label2idx_path)

    # Print Size of vocabulary and label vocabulary
    print("Vocabulary Sizes:",
          f"Input: {len(word2idx)}",
          f"POS: {len(pos2idx)}",
          f"Predicates: {len(predicate2idx)}",
          f"Output: {len(label2idx)}", sep='\n')

    training_set.encode_dataset(word2idx, label2idx, pos2idx, predicate2idx)
    # pprint(training_set.encoded_data[0])
