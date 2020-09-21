class HyperParameters:
    def __init__(self, model_name_, vocab, label_vocab, embeddings_,
                 batch_size_, pos_, predicates_, pos_embeddings_):
        self.model_name = model_name_
        self.vocab_size = len(vocab) if vocab else "Using Pretrained Model's Tokenizer"
        self.num_classes = len(label_vocab)
        self.pos_vocab_size = len(pos_)
        self.predicates_vocab_size = len(predicates_)
        self.hidden_dim = 256
        self.bidirectional = True
        self.embedding_dim = 300
        self.num_layers = 2
        self.dropout = 0.4
        self.embeddings = embeddings_
        self.pos_embeddings = pos_embeddings_
        self.batch_size = batch_size_

    def _print_info(self):
        print("========== Hyperparameters ==========",
              f"Name: {self.model_name.replace('_', ' ')}",
              f"Vocab Size: {self.vocab_size}",
              f"Tags Size: {self.num_classes}",
              f"POS Vocab Size: {self.pos_vocab_size}",
              f"Predicates Vocab Size: {self.predicates_vocab_size}",
              f"Embeddings Dim: {self.embedding_dim}",
              f"Hidden Size: {self.hidden_dim}",
              f"BiLSTM: {self.bidirectional}",
              f"Layers Num: {self.num_layers}",
              f"Dropout: {self.dropout}",
              f"Pretrained_embeddings: {False if self.embeddings is None else True}",
              f"PoS Pretrained_embeddings: {False if self.pos_embeddings is None else True}",
              f"Batch Size: {self.batch_size}", sep='\n')
