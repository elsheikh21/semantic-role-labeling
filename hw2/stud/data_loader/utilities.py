import pickle


def save_pickle(save_to, save_what):
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(load_from):
    with open(load_from, 'rb') as f:
        return pickle.load(f)
