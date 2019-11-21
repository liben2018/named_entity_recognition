from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval, ensemble_evaluate


def main():
    """Training model and evaluating results!"""
    # selecting model
    do_hmm_in_main = False
    do_crf_in_main = False
    do_bilstm_in_main = False
    do_bilstmcrf_in_main = True
    do_ensemble_in_main = False
    ensemble_model_list = []

    # Data
    print("Reading data:")
    ner_data_dir = "./FA_NER_Data_IOB"
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=ner_data_dir)
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir=ner_data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=ner_data_dir)
    print("len(train_word_lists):", len(train_word_lists))
    print("len(word2id=vocab):", len(word2id))

    if do_hmm_in_main:
        # Training and Evaluating HMM model
        print("Training and Evaluating HMM model:")
        hmm_pred = hmm_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists),
            word2id,
            tag2id
        )
        ensemble_model_list.append(hmm_pred)

    if do_crf_in_main:
        # Training and evaluating CRF model
        print("Training and evaluating CRF model:")
        crf_pred = crf_train_eval(
            (train_word_lists, train_tag_lists),
            (test_word_lists, test_tag_lists)
        )
        ensemble_model_list.append(crf_pred)

    if do_bilstm_in_main:
        # Training and evaluating BI-LSTM model
        print("Training and evaluating Bi-LSTM model:")
        # We need to put 'PAD' and 'UNK' in word2id and tag2id, when we train LSTM model.
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        lstm_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            bilstm_word2id, bilstm_tag2id,
            crf=False
        )
        ensemble_model_list.append(lstm_pred)

    if do_bilstmcrf_in_main:
        # Training and evaluating Bi-LSTM+CRF model
        print("Training and evaluating Bi-LSTM-CRF model:")
        # We need to add <start> and <end>, when we use lstm model with CRF (will be used during decoder processing).
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        # data processing
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)
        lstmcrf_pred = bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            crf_word2id, crf_tag2id,
            remove_O=False,
            reload_model=True
        )
        ensemble_model_list.append(lstmcrf_pred)

    if do_ensemble_in_main:
        ensemble_evaluate(ensemble_model_list, test_tag_lists)


if __name__ == "__main__":
    main()
