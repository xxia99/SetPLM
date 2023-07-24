"""
Implementation of "Attention is All You Need"
"""
# from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from pykp.modules.multi_head_attn import MultiHeadAttention

class TransformerSeq2SeqEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Self-Attention的Layer，
        :param int d_model: input和output的输出维度
        :param int n_head: 多少个head，每个head的维度为d_model/n_head
        :param int dim_ff: FFN的维度大小
        :param float dropout: Self-attention和FFN的dropout大小，0表示不drop
        """
        super(TransformerSeq2SeqEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch x src_seq x d_model
        :param mask: batch x src_seq，为0的地方为padding
        :return:
        """
        # attention
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              key_mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x

class TransformerSeq2SeqEncoder(nn.Module):
    def __init__(self, embed, pos_embed=None, num_layers=6, d_model=512, n_head=8, dim_ff=2048, dropout=0.1, plm=None, tk=None, idx2word={},device="cpu", encoder=None):
        """
        基于Transformer的Encoder
        :param embed: encoder输入token的embedding
        :param nn.Module pos_embed: position embedding
        :param int num_layers: 多少层的encoder
        :param int d_model: 输入输出的维度
        :param int n_head: 多少个head
        :param int dim_ff: FFN中间的维度大小
        :param float dropout: Attention和FFN的dropout大小
        """
        super(TransformerSeq2SeqEncoder, self).__init__()
        self.embed = embed
        self.embed_scale = math.sqrt(d_model)
        self.pos_embed = pos_embed
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqEncoderLayer(d_model, n_head, dim_ff, dropout)
                                           for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

        if plm:
            if encoder:
                self.plm = encoder
                for param in self.plm.parameters():
                    param.requires_grad = False
                self.tk = tk
                self.bert2encoder = torch.nn.Linear(768, 512)
                self.idx2word = idx2word
                self.device = device
            else:
                self.plm = None
                self.tk = None

    @classmethod
    def from_opt(cls, opt, embed, pos_embed):
        return cls(embed,
                   pos_embed,
                   num_layers=opt.enc_layers,
                   d_model=opt.d_model,
                   n_head=opt.n_head,
                   dim_ff=opt.dim_ff,
                   dropout=opt.dropout,
                   plm=opt.plm,
                   tk=opt.tk,
                   encoder=opt.encoder,
                   idx2word=opt.vocab["idx2word"],
                   device=opt.device)

    def forward(self, src, src_lens, src_mask):
        """
        :param tokens: batch x max_len
        :param seq_len: [batch]
        :return: bsz x max_len x d_model, bsz x max_len(为0的地方为padding)
        """
        # print("encoder: ", self.plm)
        ###### use plm encoder ######
        if self.plm:
            src_temp = src.tolist()
            input_ids = []
            for s_t in src_temp:
                s_tks = [self.idx2word[idx] for idx in s_t]
                s_new = [self.tk.convert_tokens_to_ids(word) for word in s_tks]
                input_ids.append(s_new[:512])
            input_ids = torch.tensor(input_ids).to(self.device)
            # input_ids = self.embed(input_ids) * self.embed_scale
            # print("encoder: ", self.plm)
            outputs = self.plm(input_ids=input_ids, attention_mask=src_mask)
            x1 = outputs.last_hidden_state
            x1 = self.bert2encoder(x1)
            return x1

        ###### use original encoder ######
        x = self.embed(src) * self.embed_scale  # batch, seq, dim
        batch_size, max_src_len, _ = x.size()
        device = x.device
        if self.pos_embed is not None:
            position = torch.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layer_stacks:
            x = layer(x, src_mask)

        x = self.layer_norm(x)

        return x
