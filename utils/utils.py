import pickle


def merge_maps(dict1, dict2):
    """Incorporate 2 word2id or 2 tag2id!"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM model: in training phase, need to adding <PAD> and <UNK> in word2id and tag2id.
# Furthermore, for LSTM with CRF, <start> and <end> is also needed for decoding.
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # For Bi-LSTM with CRF, need to adding <start> and <end> tokens!
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # don't need to add <end> token for test data!
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
