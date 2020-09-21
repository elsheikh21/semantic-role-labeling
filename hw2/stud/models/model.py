import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
from transformers import AutoModel


# try:
#     from torchcrf import CRF
# except ModuleNotFoundError:
#     os.system('pip install pytorch-crf')
#     from torchcrf import CRF


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.name = hparams.model_name

        self.word_embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        if hparams.embeddings is not None:
            self.word_embedding.weight.data.copy_(hparams.embeddings)
        self.dropout = nn.Dropout(hparams.dropout)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            batch_first=True,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, x):
        # [Samples_Num, Seq_Len]
        embeddings = self.word_embedding(x)
        # [Samples_Num, Seq_Len]
        embeddings_ = self.dropout(embeddings)
        # [Samples_Num, Seq_Len]
        o, _ = self.lstm(embeddings_)
        o = self.lstm_dropout(o)
        # [Samples_Num, Seq_Len, Tags_Num]
        logits = self.classifier(o)
        # [Samples_Num, Seq_Len]
        return logits

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def prepare_for_predict(self, sentence):
        """
        Helper method to prepare (n) data_x, where n is the
        number of predicates
        """
        data_x, predicate_indices = [], []
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        if num_predicates == 0:
            return sentence.get('lemmas')[:]
        for i in range(len(predicates)):
            if predicates[i] != '_':
                sentence_ = sentence.get('lemmas')[:]
                sentence_[i] = '<PREDICATE>'
                predicate_indices.append(i)
                data_x.append(sentence_)
        return data_x, predicate_indices

    def _predict_sentence(self, sentence, word2idx, label2idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        input_ = []
        if num_predicates != 0:
            test_x, pred_indices = self.prepare_for_predict(sentence)
            for sentence_ in test_x:
                input_.append(torch.LongTensor(
                    [word2idx.get(word, 1) for word in sentence_]).to(device))
            input_ = torch.stack(input_)

            self.eval()
            with torch.no_grad():
                logits = self(input_)
                predictions = torch.argmax(logits, -1)  # .view(-1)
                return predictions, pred_indices
        else:
            return {}, None

    def print_summary(self, show_weights=False, show_parameters=False):
        """
        Summarizes torch model by showing trainable parameters and weights.
        """
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        print(f"Params #: {'{:,}'.format(num_params)}")
        print('==================================================')


class MultiInputModel(nn.Module):
    def __init__(self, hparams):
        super(MultiInputModel, self).__init__()
        self.name = hparams.model_name

        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        if hparams.embeddings is not None:
            self.word_embedding.weight.data.copy_(hparams.embeddings)
        self.dropout = nn.Dropout(hparams.dropout)

        self.predicates_embedding = nn.Embedding(hparams.predicates_vocab_size, hparams.embedding_dim, padding_idx=0)

        self.predicates_dropout = nn.Dropout(hparams.dropout)

        self.pos_embedding = nn.Embedding(hparams.pos_vocab_size, hparams.embedding_dim, padding_idx=0)

        self.pos_dropout = nn.Dropout(hparams.dropout)

        self.lstm = nn.LSTM(hparams.embedding_dim * 3, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            batch_first=True,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, tokens_seq, predicates_seq, pos_seq):
        # inputs shape: [batch size, maximum sequence length]
        embeddings = self.word_embedding(tokens_seq)
        embeddings_ = self.dropout(embeddings)
        # Embeddings shape: [batch size, maximum sequence length, embeddings size]

        predicates_embeddings = self.predicates_embedding(predicates_seq)
        predicates_embeddings_ = self.predicates_dropout(predicates_embeddings)
        # Embeddings shape: [batch size, maximum sequence length, embeddings size]

        pos_embeddings = self.pos_embedding(pos_seq)
        pos_embeddings_ = self.pos_dropout(pos_embeddings)

        embeds_ = torch.cat((embeddings_, predicates_embeddings_, pos_embeddings_), dim=2)
        # embeds_ = torch.cat((embeddings_, predicates_embeddings_), dim=2)
        # Embeddings Concatenated shape
        # [batch size, maximum sequence length, embeddings size * 2]

        # LSTM Waits shapes [embeddings size * 2, max sequence length, hidden_size]
        o, _ = self.lstm(embeds_)
        o = self.lstm_dropout(o)
        # LSTM Out shapes [Batch Size, max sequence length, hidden_size]
        logits = self.classifier(o)
        return logits

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path, _device):
        state_dict = torch.load(path) if _device == 'cuda' else torch.load(path, map_location=torch.device(_device))
        self.load_state_dict(state_dict)

    def prepare_for_predict(self, sentence):
        data_x, predicate_indices = [], []
        predicates_x, pos_x = [], []
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        if num_predicates == 0:
            return sentence.get('lemmas')[:], _
        for i in range(len(predicates)):
            if predicates[i] != '_':
                sentence_lemmas_ = sentence.get('lemmas')[:]
                predicates_sentence = sentence.get('predicates')[:]
                pos_sentence = sentence.get('pos_tags')[:]

                verb_atlas_grp = predicates_sentence[i]
                predicates_sentence_ = ["_"] * len(predicates_sentence)
                predicates_sentence_[i] = verb_atlas_grp

                data_x.append(sentence_lemmas_)
                predicate_indices.append(i)
                predicates_x.append(predicates_sentence_)
                pos_x.append(pos_sentence)
        return data_x, predicates_x, pos_x, predicate_indices

    def _predict_sentence(self, sentence, word2idx, label2idx, pos2idx, predicates2idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        input_, input_predicate, input_pos = [], [], []
        if num_predicates != 0:
            test_x, predicates_x, pos_x, pred_indices = self.prepare_for_predict(sentence)
            for sentence_, sentence_predicate, sentence_pos in zip(test_x, predicates_x, pos_x):
                input_.append(torch.LongTensor([word2idx.get(word, 1) for word in sentence_]).to(device))
                input_predicate.append(
                    torch.LongTensor([predicates2idx.get(word, 1) for word in sentence_predicate]).to(device))
                input_pos.append(torch.LongTensor([pos2idx.get(pos_tag, 1) for pos_tag in sentence_pos]).to(device))
            input_, input_predicate, input_pos = torch.stack(input_), torch.stack(input_predicate), torch.stack(
                input_pos)

            # self.eval()
            # with torch.no_grad():
            logits = self(input_, input_predicate, input_pos)
            predictions = torch.argmax(logits, -1)  # .view(-1)
            return predictions, pred_indices
        else:
            return {}, None

    def print_summary(self, show_weights=False, show_parameters=False):
        """
        Summarizes torch model by showing trainable parameters and weights.
        """
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Params #: {'{:,}'.format(num_params)}")
        print('==================================================')


class RoBERTaMultiInputModel(nn.Module):
    def __init__(self, hparams, freeze_model=True):
        super(RoBERTaMultiInputModel, self).__init__()
        self.name = hparams.model_name
        self.roberta = AutoModel.from_pretrained('roberta-base')
        if freeze_model:  # Freeze model Layer
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(hparams.dropout)

        self.predicates_embedding = nn.Embedding(hparams.predicates_vocab_size, hparams.embedding_dim, padding_idx=0)
        self.predicates_dropout = nn.Dropout(hparams.dropout)

        self.pos_embedding = nn.Embedding(hparams.pos_vocab_size, hparams.embedding_dim, padding_idx=0)
        self.pos_dropout = nn.Dropout(hparams.dropout)

        # FIXME: CAN'T COPY WEIGHTS DUE TO MISMATCH FROM POS2VEC SHAPE
        # WAITING 47 AND LOADING 41 FROM POS_Embeddings
        # if hparams.pos_embeddings is not None:
        # self.pos_embedding.weight.data.copy_(hparams.pos_embeddings)

        # Take into consideration pos & predicate embeddings both of 300D and Roberta Hidden Dim
        lstm_input_dim = (hparams.embedding_dim * 2) + (self.roberta.config.hidden_size)
        self.lstm = nn.LSTM(lstm_input_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            batch_first=True,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        # self.crf = CRF(hparams.num_classes, batch_first=True)

    def forward(self, tokens_seq, atten_masks, predicates_seq, pos_seq):
        last_hidden_layer, _ = self.roberta(input_ids=tokens_seq,
                                            attention_mask=atten_masks)
        embeddings_ = self.dropout(last_hidden_layer)

        predicates_embeddings = self.predicates_embedding(predicates_seq)
        predicates_embeddings_ = self.predicates_dropout(predicates_embeddings)

        pos_embeddings = self.pos_embedding(pos_seq)
        pos_embeddings_ = self.pos_dropout(pos_embeddings)

        non_zeros = (tokens_seq != 0).to("cuda", dtype=torch.uint8).sum(dim=1).tolist()

        reconstructed = []
        for idx in range(len(non_zeros)):
            reconstructed.append(last_hidden_layer[idx, 1:non_zeros[idx] - 3, :])
        padded_again = torch.nn.utils.rnn.pad_sequence(reconstructed, batch_first=True, padding_value=1)

        embeds_ = torch.cat((padded_again, predicates_embeddings_, pos_embeddings_), dim=-1)

        o, _ = self.lstm(embeds_)
        o = self.lstm_dropout(o)
        logits = self.classifier(o)
        return logits

    # def log_probs(self, x, tags, mask, pos, predicates):
    #     emissions = self(x, mask, predicates, pos)
    #     return self.crf(emissions, tags, mask=mask)

    # def predict(self, x, mask, pos, predicates):
    #     emissions = emissions = self(x, mask, predicates, pos)
    #     return self.crf.decode(emissions)

    def _save(self, model_path):
        torch.save(self, model_path)
        model_checkpoint = model_path.replace('.pt', '.pth')
        torch.save(self.state_dict(), model_checkpoint)

    def _load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def prepare_for_predict(self, sentence):
        data_x, predicate_indices = [], []
        predicates_x, pos_x = [], []
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        if num_predicates == 0:
            return sentence.get('lemmas')[:], _
        for i in range(len(predicates)):
            if predicates[i] != '_':
                sentence_lemmas_ = sentence.get('lemmas')[:]
                predicates_sentence = sentence.get('predicates')[:]
                pos_sentence = sentence.get('pos_tags')[:]

                verb_atlas_grp = predicates_sentence[i]
                predicates_sentence_ = ["_"] * len(predicates_sentence)
                predicates_sentence_[i] = verb_atlas_grp
                sentence_lemmas_.append('[SEP]')
                sentence_lemmas_.append(str(verb_atlas_grp))
                sentence_lemmas_.append('[SEP]')

                data_x.append(sentence_lemmas_)
                predicate_indices.append(i)
                predicates_x.append(predicates_sentence_)
                pos_x.append(pos_sentence)
        return data_x, predicates_x, pos_x, predicate_indices

    def _predict_sentence(self, sentence, word2idx, label2idx, pos2idx, predicates2idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        predicates = sentence.get('predicates')
        num_predicates = len(predicates) - predicates.count('_')
        input_, input_mask, input_predicate, input_pos = [], [], [], []
        if num_predicates != 0:
            test_x, predicates_x, pos_x, pred_indices = self.prepare_for_predict(sentence)
            for sentence_, sentence_predicate, sentence_pos in zip(test_x, predicates_x, pos_x):
                encoded_dict = tokenizer.encode_plus(text=sentence_, add_special_tokens=True, return_tensors='pt')
                # input_.append(torch.LongTensor([word2idx.get(word, 1) for word in sentence_]).to(device))
                input_ids = torch.squeeze(encoded_dict['input_ids']).to(device)
                input_.append(input_ids)
                input_mask.append((input_[-1] != 1).to(device, dtype=torch.uint8))
                input_predicate.append(
                    torch.LongTensor([predicates2idx.get(word, 1) for word in sentence_predicate]).to(device))
                input_pos.append(torch.LongTensor([pos2idx.get(pos_tag, 1) for pos_tag in sentence_pos]).to(device))
            lst_input_, lst_input_mask, lst_input_predicate, lst_input_pos = torch.stack(input_), torch.stack(
                input_mask), torch.stack(input_predicate), torch.stack(input_pos)

            self.eval()
            with torch.no_grad():
                logits = self.predict(lst_input_, lst_input_mask, lst_input_pos, lst_input_predicate)
                predictions = torch.argmax(logits, -1)  # .view(-1)
                return predictions, pred_indices
        else:
            return {}, None

    def print_summary(self, show_weights=False, show_parameters=False):
        """
        Summarizes torch model by showing trainable parameters and weights.
        """
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Params #: {'{:,}'.format(num_params)}")
        print('==================================================')
