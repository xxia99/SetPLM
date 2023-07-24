import torch.nn as nn

import pykp.utils.io as io
from pykp.decoder.transformer import TransformerSeq2SeqDecoder
from pykp.encoder.transformer import TransformerSeq2SeqEncoder
from pykp.modules.position_embed import get_sinusoid_encoding_table
from transformers import AutoModel, BertTokenizer


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()

        if opt.plm:
            # words = list(opt.vocab["word2idx"].keys())
            # self.plm, self.tk = self.load_plm(words, opt.plm)
            #
            # new_word2idx = self.tk.vocab.copy()
            # new_word2idx.update(self.tk.added_tokens_encoder)
            # opt.vocab["word2idx"] = new_word2idx
            #
            # new_idx2word = self.tk.ids_to_tokens.copy()
            # new_idx2word.update(self.tk.added_tokens_decoder)
            # opt.vocab["idx2word"] = new_idx2word

            # opt.plm = self.plm
            # opt.tk = self.tk
            # opt.vocab_size = len(new_idx2word.items())
            self.plm = opt.plm
            self.tk = opt.tk
            pos_num = opt.vocab_size
            # for param in self.plm.parameters():
            #     param.requires_grad = False
        else:
            opt.plm = None
            opt.tk = None
            pos_num = 3000

        embed = nn.Embedding(opt.vocab_size, opt.word_vec_size, opt.vocab["word2idx"][io.PAD_WORD])
        self.init_emb(embed)
        pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(pos_num, opt.word_vec_size, padding_idx=opt.vocab["word2idx"][io.PAD_WORD]),
            freeze=True)
        self.encoder = TransformerSeq2SeqEncoder.from_opt(opt, embed, pos_embed)
        self.decoder = TransformerSeq2SeqDecoder.from_opt(opt, embed, pos_embed)
        self.device = opt.device

    def load_plm(self, words, plm):
        model = plm
        tokenizer = BertTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)
        special_tokens = {'pad_token': '<pad>',
                          # 'unk_token': '<unk>',
                          'bos_token': '<bos>',
                          'eos_token': '<eos>',
                          'sep_token': '<sep>',
                          'additional_special_tokens': ['<digit>', '<peos>', '<null>', '<unk>']
                          }
        num_add_special = tokenizer.add_special_tokens(special_tokens)
        num_added_toks = tokenizer.add_tokens(words)  # 返回一个数，表示加入的新词数量，在这里是2
        # print(num_added_toks)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def init_emb(self, embed):
        """Initialize weights."""
        initrange = 0.1
        embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, input_tgt, src_oov, max_num_oov, src_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        # Encoding
        memory_bank = self.encoder(src, src_lens, src_mask)
        state = self.decoder.init_state(memory_bank, src_mask)
        decoder_dist_all, attention_dist_all = self.decoder(input_tgt, state, src_oov, max_num_oov)
        return decoder_dist_all, attention_dist_all


