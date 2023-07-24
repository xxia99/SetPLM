import logging

import torch
from torch.utils.data import DataLoader

from pykp.utils.io import KeyphraseDataset
from transformers import AutoModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel, T5Model


def load_plm(words, opt):
    model = opt.plm
    if "t5" in model:
        tokenizer = T5Tokenizer.from_pretrained(model)
        if opt.encoder:
            opt.encoder = T5EncoderModel.from_pretrained(model)
        # model = T5ForConditionalGeneration.from_pretrained(model, output_attentions=True)
        model = T5Model.from_pretrained(model, output_attentions=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
        opt.encoder = model
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
    if opt.encoder:
        opt.encoder.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % opt.vocab)
    vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')
    # assign vocab to opt
    if opt.plm:
        opt.plm_name = opt.plm
        words = list(vocab["word2idx"].keys())
        plm, tk = load_plm(words, opt)
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
        print("dataloader vocab: ", len(new_word2idx.items()))
        print("dataloader peos id: ", new_word2idx['<peos>'])


        opt.plm = plm
        opt.tk = tk
        opt.vocab_size = len(new_idx2word.items())
    else:
        opt.plm = None
        opt.tk = None
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab["word2idx"]))
    # opt.vocab_size = len(vocab["word2idx"])
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return vocab


def build_data_loader(data, opt, shuffle=True, load_train=True):
    keyphrase_dataset = KeyphraseDataset.build(examples=data, opt=opt, load_train=load_train)
    if not opt.one2many:
        collect_fn = keyphrase_dataset.collate_fn_one2one
    elif opt.fix_kp_num_len:
        collect_fn = keyphrase_dataset.collate_fn_fixed_tgt
    else:
        collect_fn = keyphrase_dataset.collate_fn_one2seq

    data_loader = DataLoader(dataset=keyphrase_dataset, collate_fn=collect_fn, num_workers=opt.batch_workers,
                             batch_size=opt.batch_size, shuffle=shuffle)
    return data_loader


def load_data_and_vocab(opt, load_train=True):
    vocab = load_vocab(opt)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)
    if opt.one2many:
        data_path = opt.data + '/%s.one2many.pt'
    else:
        data_path = opt.data + '/%s.one2one.pt'

    if load_train:
        # load training dataset
        train_data = torch.load(data_path % "train", 'wb')
        train_loader = build_data_loader(data=train_data, opt=opt, shuffle=True, load_train=True)
        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

        # load validation dataset
        valid_data = torch.load(data_path % "valid", 'wb')
        valid_loader = build_data_loader(data=valid_data,  opt=opt, shuffle=False, load_train=True)
        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, vocab
    else:
        test_data = torch.load(data_path % "test", 'wb')
        test_loader = build_data_loader(data=test_data, opt=opt, shuffle=False, load_train=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))
        return test_loader, vocab
