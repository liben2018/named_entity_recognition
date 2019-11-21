import time
from collections import Counter

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists, load_model
from evaluating import Metrics, get_ner_fmeasure, results_as_entities
from models.config import LSTMConfig


def results_print(test_tag_lists, pred_tag_lists, remove_O=False):
    use_original_metrix = False
    use_entity_level_metrix = False  # Not work for char-level labels to get entity-level
    use_seqeval = True
    if use_original_metrix:
        metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()
    elif use_entity_level_metrix:
        get_ner_fmeasure(test_tag_lists, pred_tag_lists, label_type="BIOES")
    elif use_seqeval:
        results = seqeval_output(test_tag_lists, pred_tag_lists)
        print(results)
    else:
        from sklearn.metrics import classification_report, confusion_matrix
        test_tag_lists = flatten_lists(test_tag_lists)
        pred_tag_lists = flatten_lists(pred_tag_lists)
        # delete "O" labels
        delete_other_label = remove_O
        if delete_other_label:
            length = len(test_tag_lists)
            other_tag_indices = [i for i in range(length) if test_tag_lists[i] == 'O']
            test_tag_lists = [tag for i, tag in enumerate(test_tag_lists) if i not in other_tag_indices]
            pred_tag_lists = [tag for i, tag in enumerate(pred_tag_lists) if i not in other_tag_indices]
        print(classification_report(test_tag_lists, pred_tag_lists))
        # print(confusion_matrix(test_tag_lists, pred_tag_lists))


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    # data
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    # training
    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    save_model(hmm_model, "./ckpts/hmm.pkl")

    # evaluating
    pred_tag_lists = hmm_model.test(test_word_lists, word2id, tag2id)
    results_print(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    return pred_tag_lists


def crf_train_eval(train_data, test_data, remove_O=False):

    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    model_file = "./ckpts/crf.pkl"
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, model_file)
    # crf_model = load_model(model_file)

    pred_tag_lists = crf_model.test(test_word_lists)
    results_print(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data, word2id, tag2id,
                          crf=True,
                          remove_O=False,
                          reload_model=False):
    # data
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    # training
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    # get model_file
    if crf:
        model_name = "bilstm_crf"
    else:
        model_name = "bilstm"
    emb_size = LSTMConfig.emb_size
    hidden_size = LSTMConfig.hidden_size
    model_file = "./ckpts/" + model_name + '_' + str(emb_size) + '_' + str(hidden_size) + ".pkl"

    if reload_model:
        # reload trained model!
        bilstm_model = load_model(model_file)
    else:
        # train and save model!
        bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
        bilstm_model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id)
        save_model(bilstm_model, model_file)  # re-thinking when to save the model? after valid for each epoch?
    print("Training finished, taken {} seconds!".format(int(time.time()-start)))
    print("Evaluating {} model:".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(test_word_lists, test_tag_lists, word2id, tag2id)
    results_print(test_tag_lists, pred_tag_lists, remove_O=remove_O)

    return pred_tag_lists


def ensemble_evaluate(results, targets, remove_O=False):
    """model ensemble"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble results for {} models:".format(len(results)))
    results_print(targets, pred_tags, remove_O=remove_O)
