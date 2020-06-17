import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch

from model import Model
from stud.data_loader import JSONDataParser
from stud.models import HyperParameters, MultiInputModel
from stud.utilities import (configure_seed_logging, load_pickle,
                            load_pretrained_embeddings)


def build_model_34(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(device_=device)


def build_model_234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


def build_model_1234(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    # return Baseline(return_predicates=True)
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, return_predicates=False):
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence['pos_tags']:
            prob = self.baselines['predicate_identification'][pos]['positive'] / self.baselines['predicate_identification'][pos]['total']
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(zip(sentence['lemmas'], predicate_identification)):
            if not is_predicate or lemma not in self.baselines['predicate_disambiguation']:
                predicate_disambiguation.append('_')
            else:
                predicate_disambiguation.append(self.baselines['predicate_disambiguation'][lemma])
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence['dependency_relations']:
            prob = self.baselines['argument_identification'][dependency_relation]['positive'] / self.baselines['argument_identification'][dependency_relation]['total']
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(sentence['dependency_relations'], argument_identification):
            if not is_argument:
                argument_classification.append('_')
            else:
                argument_classification.append(self.baselines['argument_classification'][dependency_relation])

        if self.return_predicates:
            return {
                'predicates': predicate_disambiguation,
                'roles': {i: argument_classification for i in predicate_indices},
            }
        else:
            return {'roles': {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path='data/baselines.json'):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):
    def __init__(self, device_):
        self.device = device_

        MODEL_FOLDER = os.path.join(os.getcwd(), 'model')
        self.word2idx = load_pickle(os.path.join(MODEL_FOLDER, 'word_stoi.pkl'))
        self.idx2label = load_pickle(os.path.join(MODEL_FOLDER, 'idx2label.pkl'))
        self.label2idx = load_pickle(os.path.join(MODEL_FOLDER, 'label2idx.pkl'))
        self.pos2idx = load_pickle(os.path.join(MODEL_FOLDER, 'pos_stoi.pkl'))
        self.predicate2idx = load_pickle(os.path.join(MODEL_FOLDER, 'predicate_stoi.pkl'))
        self.pretrained_embeddings = torch.from_numpy(np.load(os.path.join(MODEL_FOLDER, 'vocab_embeddings_vector.npy')))

        self._build_model()

    def _build_model(self):
        hp = HyperParameters(model_name_='Multi_Input_Stacked_BiLSTM_Model', vocab=self.word2idx,
                             label_vocab=self.idx2label, embeddings_=self.pretrained_embeddings,
                             batch_size_=128, pos_=self.pos2idx, predicates_=self.predicate2idx)

        model_path = os.path.join(os.getcwd(), 'model',
                                  'MultInput_Stacked_BiLSTM_Fasttext_ClipGrad_ckpt_best.pth')
        self.model = MultiInputModel(hp).to(self.device)
        self.model._load(path=model_path, _device=self.device)
        self.model.eval()

    def predict(self, sentence):
        with torch.no_grad():
            logits, predicates_indices = self.model._predict_sentence(sentence, self.word2idx,
                                                                      self.label2idx, self.pos2idx,
                                                                      self.predicate2idx)
            if logits != {}:
                prediction = JSONDataParser.decode_predictions(logits, self.idx2label)
                return { "roles" : dict(zip(predicates_indices, prediction)) }
            else:
                return { "roles" : dict([]) }
