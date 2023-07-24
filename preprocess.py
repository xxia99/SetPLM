import argparse
import logging
import os
import json
from collections import Counter
import torch

import config
import pykp.utils.io as io
from utils.functions import read_src_and_trg_files
from transformers import AutoModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel


def load_plm(words, plm):
    model = plm
    if "t5" in model:
        tokenizer = T5Tokenizer.from_pretrained(model)
        model = T5EncoderModel.from_pretrained(model)
    else:
        tokenizer = BertTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
    special_tokens = {'pad_token': '<pad>',
                      # 'unk_token': '<unk>',
                      'bos_token': '<bos>',
                      'eos_token': '<eos>',
                      'sep_token': '<sep>',
                      'cls_token': '<cls>',
                      'additional_special_tokens': ['<digit>', '<peos>', '<null>', '<unk>']
                      }
    num_add_special = tokenizer.add_special_tokens(special_tokens)
    num_added_toks = tokenizer.add_tokens(words)  # 返回一个数，表示加入的新词数量，在这里是2
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def build_vocab(tokenized_src_trg_pairs):
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists, key_word_list in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        token_freq_counter.update(key_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    special_tokens = [io.PAD_WORD, io.UNK_WORD, io.BOS_WORD, io.EOS_WORD, io.SEP_WORD, io.PEOS_WORD,
                      io.NULL_WORD]
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    vocab = {"word2idx": word2idx, "idx2word": idx2word, "counter": token_freq_counter}
    return vocab

# def add_kws(opt, tokenized_train_pairs):
#     with open(opt.train_kws) as f_kws:
#         lines = json.load(f_kws)
#         kwords = lines["keywords"]
#         for i in range(len(tokenized_train_pairs)):
#             pairs_tmp = list(tokenized_train_pairs[i])
#             pairs_tmp.append(kwords[i])
#             new_pairs = tuple(pairs_tmp)

def main(opt):
    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    tokenized_train_pairs = read_src_and_trg_files(opt.train_src, opt.train_trg, opt.train_kws, is_train=True,
                                                   remove_title_eos=opt.remove_title_eos, prompt=opt.prompt)
    # tokenized_train_pairs = add_kws(opt, tokenized_train_pairs)

    tokenized_valid_pairs = read_src_and_trg_files(opt.valid_src, opt.valid_trg, opt.valid_kws, is_train=False,
                                                   remove_title_eos=opt.remove_title_eos, prompt=opt.prompt)

    vocab = build_vocab(tokenized_train_pairs)
    opt.vocab = vocab
    if opt.plm:
        opt.plm_name = opt.plm
        words = list(vocab["word2idx"].keys())[:50000]
        plm, tk = load_plm(words, opt.plm)
        if "t5" in opt.plm_name:
            new_word2idx = tk.get_vocab().copy()
            # new_word2idx.update(tk.added_tokens_encoder)
            # test_id = new_word2idx["<digit>"]
            new_idx2word = {v: k for k, v in new_word2idx.items()}
        else:
            new_idx2word = tk.ids_to_tokens.copy()
            new_word2idx = tk.vocab.copy()
            new_word2idx.update(tk.added_tokens_encoder)
            new_idx2word.update(tk.added_tokens_decoder)

        vocab["word2idx"] = new_word2idx
        vocab["idx2word"] = new_idx2word
        print("preprocess vocab: ", len(new_word2idx.items()))
        print("preprocess peos id: ", new_word2idx['<peos>'])

        opt.plm = plm
        opt.tk = tk
        opt.vocab_size = len(new_idx2word.items())
    else:
        opt.plm = None
        opt.tk = None
    logging.info("Dumping dict to disk: %s" % opt.save_data_dir + '/vocab.pt')
    torch.save(vocab, open(opt.save_data_dir + '/vocab.pt', 'wb'))

    if not opt.one2many:
        # saving  one2one datasets
        train_one2one = io.build_dataset(tokenized_train_pairs, opt, mode='one2one')
        logging.info("Dumping train one2one to disk: %s" % (opt.save_data_dir + '/train.one2one.pt'))
        torch.save(train_one2one, open(opt.save_data_dir + '/train.one2one.pt', 'wb'))
        len_train_one2one = len(train_one2one)
        del train_one2one

        valid_one2one = io.build_dataset(tokenized_valid_pairs, opt, mode='one2one')
        logging.info("Dumping valid to disk: %s" % (opt.save_data_dir + '/valid.one2one.pt'))
        torch.save(valid_one2one, open(opt.save_data_dir + '/valid.one2one.pt', 'wb'))

        logging.info('#pairs of train_one2one  = %d' % len_train_one2one)
        logging.info('#pairs of valid_one2one  = %d' % len(valid_one2one))
    else:
        # saving  one2many datasets
        train_one2many = io.build_dataset(tokenized_train_pairs, opt, mode='one2many')
        logging.info("Dumping train one2many to disk: %s" % (opt.save_data_dir + '/train.one2many.pt'))
        torch.save(train_one2many, open(opt.save_data_dir + '/train.one2many.pt', 'wb'))
        len_train_one2many = len(train_one2many)
        del train_one2many

        valid_one2many = io.build_dataset(tokenized_valid_pairs, opt, mode='one2many')
        logging.info("Dumping valid to disk: %s" % (opt.save_data_dir + '/valid.one2many.pt'))
        torch.save(valid_one2many, open(opt.save_data_dir + '/valid.one2many.pt', 'wb'))

        logging.info('#pairs of train_one2many = %d' % len_train_one2many)
        logging.info('#pairs of valid_one2many = %d' % len(valid_one2many))
    logging.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    opt = parser.parse_args()

    logging = config.init_logging(log_file=opt.log_path + "/output.log", stdout=True)

    #opt.save_data_dir = "./processed_data_new"
    if not opt.one2many:
        test_exists = os.path.join(opt.save_data_dir, "train.one2one.pt")
    else:
        test_exists = os.path.join(opt.save_data_dir, "train.one2many.pt")
    # if os.path.exists(test_exists):
    #     logging.info("file exists %s, exit! " % test_exists)
    #     exit()

    opt.train_src = opt.data_dir + '/train_src.txt'
    opt.train_trg = opt.data_dir + '/train_trg.txt'
    opt.train_kws = opt.data_dir + '/train_kws.txt'
    opt.valid_src = opt.data_dir + '/valid_src.txt'
    opt.valid_trg = opt.data_dir + '/valid_trg.txt'
    opt.valid_kws = opt.data_dir + '/valid_kws.txt'
    main(opt)
    print("Preprocessed Vocab Size: ", opt.vocab_size)
