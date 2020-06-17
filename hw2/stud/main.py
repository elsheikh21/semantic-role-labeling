import os
import random
import numpy as np
import logging
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader import JSONDataParser
from models import BaselineModel, MultiInputModel, HyperParameters
from training import Trainer, WriterTensorboardX
from evaluate import evaluate_argument_identification, evaluate_argument_classification
from utilities import configure_seed_logging, load_pretrained_embeddings


if __name__ == "__main__":
    configure_seed_logging()

    path = os.path.join(os.getcwd(), 'data', 'train.json')
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse Training Dataset
    training_set = JSONDataParser(path, device=_device)
    training_set.get_sentences()
    word2idx_path = os.path.join(os.getcwd(), 'model', 'word_stoi.pkl')
    word2idx, idx2word = JSONDataParser.build_vocabulary(
        training_set.data_x, word2idx_path)
    pos2idx_path = os.path.join(os.getcwd(), 'model', 'pos_stoi.pkl')
    pos2idx, idx2pos = JSONDataParser.build_vocabulary(
        training_set.pos_x, pos2idx_path)
    predicate2idx_path = os.path.join(
        os.getcwd(), 'model', 'predicate_stoi.pkl')
    predicate2idx, idx2predicate = JSONDataParser.build_vocabulary(
        training_set.predicates_x, predicate2idx_path)
    label2idx_path = os.path.join(os.getcwd(), 'model', 'label2idx.pkl')
    label2idx, idx2label = JSONDataParser.build_labels_vocabulary(
        training_set.data_y, label2idx_path)
    training_set.encode_dataset(word2idx, label2idx, pos2idx, predicate2idx)

    # Parse Development Dataset
    dev_path = os.path.join(os.getcwd(), 'data', 'dev.json')
    dev_set = JSONDataParser(dev_path, device=_device)
    dev_set.get_sentences()
    dev_set.encode_dataset(word2idx, label2idx, pos2idx, predicate2idx)

    # Parse Test Dataset
    test_path = os.path.join(os.getcwd(), 'data', 'test.json')
    test_set = JSONDataParser(test_path, device=_device)
    test_set.get_sentences()
    test_set.encode_dataset(word2idx, label2idx, pos2idx, predicate2idx)

    save_to = os.path.join(os.getcwd(), 'model', 'vocab_embeddings_vector.npy')
    embeddings_path = '/content/drive/My Drive/HW1_NLP/wiki.en.vec'
    pretrained_embeddings_ = load_pretrained_embeddings(embeddings_path, word2idx,
                                                        300, save_to=save_to)

    # Set Hyper-parameters
    batch_size = 128

    name_ = 'Multi Input Stacked BiLSTM Model'
    hp = HyperParameters(name_, word2idx, label2idx,
                         pretrained_embeddings_, batch_size,
                         pos2idx, predicate2idx)
    hp._print_info()

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=training_set,batch_size=batch_size,
                                collate_fn=JSONDataParser.pad_batch, shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_set, batch_size=batch_size,
                              collate_fn=JSONDataParser.pad_batch)
    test_dataset_ = DataLoader(dataset=test_set, batch_size=batch_size,
                               collate_fn=JSONDataParser.pad_batch)

    # Create and train model
    model = MultiInputModel(hp).to(_device)
    model.print_summary()

    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

    trainer = Trainer(model=model, writer=writer_,
                      loss_function=CrossEntropyLoss(ignore_index=label2idx['<PAD>']),
                      optimizer=Adam(model.parameters()), epochs=50,
                      num_classes=hp.num_classes, verbose=True)

    save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model.pt")
    _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)

    sentences, labels = test_set.sentences, test_set.labels

    predictions_ = {}
    for id_, sentence in sentences.items():
        # if id_ == batch_size: break
        logits, predicates_indices = model._predict_sentence(sentence, word2idx, label2idx, pos2idx, predicate2idx)
        if logits != {}:
            prediction = JSONDataParser.decode_predictions(logits, idx2label)
            predictions_[id_] = { "roles" : dict(zip(predicates_indices, prediction)) }
        else:
            predictions_[id_] = { "roles" : logits }

    labels_ = {}
    for id_, _ in labels.items():
        # if id_ == batch_size: break
        labels_[id_] = labels[id_]

    print("_______" * 5)
    d1 = evaluate_argument_identification(labels_, predictions_)
    print(f"Identification Recall: {d1.get('precision'):.4f}")
    print(f"Identification Precision: {d1.get('recall'):.4f}")
    print(f"Identification F1: {d1.get('f1'):.4f}")

    print("_______" * 5)

    d2 = evaluate_argument_classification(labels_, predictions_)
    print(f"Classification Recall: {d2.get('precision'):.4f}")
    print(f"Classification Precision: {d2.get('recall'):.4f}")
    print(f"Classification F1: {d2.get('f1'):.4f}")
    print("_______" * 5)