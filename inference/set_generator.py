import pykp.utils.io as io
import torch

EPS = 1e-8

class SetGenerator(object):
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_opt(cls, model, opt):
        return cls(model)

    def inference(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx, idx2word, kws, src_str_list=None):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        """
        self.model.eval()
        batch_size = src.size(0)
        max_kp_num = self.model.decoder.max_kp_num
        max_kp_len = self.model.decoder.max_kp_len
        vocab_size = self.model.decoder.vocab_size
        # print("decoder vocab: ", vocab_size)

        # Encoding
        memory_bank = self.model.encoder(src, src_lens, src_mask)
        # print("memory_bank: ", memory_bank)
        state = self.model.decoder.init_state(memory_bank, src_mask)
        # print("state: ", state)
        control_embed, kws_embed = self.model.decoder.forward_seg(state, kws)
        # print("kws_embed: ", kws_embed)
        # print("src_oov: ", src_oov)

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
        attn_dict_list = []
        decoder_score_list = []
        output_tokens = src.new_zeros(batch_size, max_kp_num, max_kp_len + 1)
        output_tokens[:, :, 0] = word2idx[io.BOS_WORD]
        # print("output_0: ", output_tokens[:, :, 0])
        for t in range(1, max_kp_len+1):
            decoder_inputs = output_tokens[:, :, :t]
            decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(vocab_size - 1), word2idx[io.UNK_WORD])
            # if t == 1:
            #     print("decode_input: ", decoder_inputs)
            decoder_dist, attn_dist = self.model.decoder(decoder_inputs, state, src_oov, max_num_oov, kws_embed, src)
            # if t == 1:
            #     print("decoder_dist: ", decoder_dist.size())
            attn_dict_list.append(attn_dist.reshape(batch_size, max_kp_num, 1, -1))
            decoder_score_list.append(decoder_dist.max(-1)[0].reshape(batch_size, max_kp_num, 1))

            _, tokens = decoder_dist.max(-1)
            output_tokens[:, :, t] = tokens
        # print("tokens: ", tokens)
        output_tokens = output_tokens[:, :, 1:].reshape(batch_size, max_kp_num*max_kp_len)[:, None]  # [batch_size, 1, max_kp_num, max_kp_len]
        attn_dicts = torch.cat(attn_dict_list, -2).reshape(batch_size, max_kp_num*max_kp_len, -1)[:, None]  # [batch_size, 1, max_kp_num, max_kp_len, max_src_len]
        decoder_scores = torch.cat(decoder_score_list, -1).reshape(batch_size, max_kp_num * max_kp_len)[:, None]

        # Extract sentences
        result_dict = {"predictions": [],
                       "attention": [],
                       "decoder_scores": [],
                       "generated_results": [],
                       "sequences_scores": []}
        # print("word2idx: ", len(word2idx))
        # print("idx2word: ", len(idx2word))


        for b in range(batch_size):
            result_dict["predictions"].append(output_tokens[b])
            result_dict["attention"].append(attn_dicts[b])
            result_dict["decoder_scores"].append(decoder_scores[b])


            # result_dict["generated_results"].append([])
            # result_dict["sequences_scores"].append([])
            # # print("src: ", src_str_list[b])
            # # print("kw: ", kws[b])

            # result_b = []
            # score_b = []
            # for ind in kws[b].tolist():
            #     kw_id = ind
            #     kw = idx2word[kw_id]
            #     # print('kw: ', kw)
            #     if ind:
            #         prompt = (f'generate a key phrase containing {kw} from <sep>').split(' ')
            #     else:
            #         prompt = (f'generate a key phrase containing {kw} from <sep>').split(' ')
            #     prompt_ids = [word2idx[word] for word in prompt]
            #     input_ids = prompt_ids + src[b].tolist()[5:]
            #     # input_words = [idx2word[idx] for idx in input_ids]
            #     input_mask = src_mask[b].tolist()[5:] + [1] * len(prompt_ids)
            #     input = torch.LongTensor(input_ids).view(1, -1)
            #     input_mask = torch.LongTensor(input_mask).view(1, -1)
            #
            #     if ind:
            #         t5_result = self.model.decoder.plm.generate(input_ids=input.to(self.model.device),
            #                                                 # src[b].view(1, -1),
            #                                                 # inputs=input,
            #                                                 attention_mask=input_mask.to(self.model.device),
            #                                                 max_length=3,
            #                                                 force_words_ids=[[kw_id]],
            #                                                 num_beams=2,
            #                                                 # temperature=0.1,
            #                                                 no_repeat_ngram_size=2,
            #                                                 output_scores=True,
            #                                                 return_dict_in_generate=True,
            #                                                 )
            #         scores = t5_result.sequences_scores
            #         score_b.append(scores.item())
            #         # result_dict["sequences_scores"][b].append(scores.item())
            #         texts = t5_result.sequences.tolist()
            #     # else:
            #     #     t5_result = self.model.decoder.plm.generate(input_ids=input.to(self.model.device),
            #     #                                                 attention_mask=input_mask.to(self.model.device),
            #     #                                                 max_length=3,
            #     #                                                 # force_words_ids=[[kw_id]],
            #     #                                                 num_beams=5,
            #     #                                                 do_sample=True,
            #     #                                                 temperature=0.9,
            #     #                                                 no_repeat_ngram_size=2,
            #     #                                                 )
            #     #     texts = t5_result.tolist()
            #     # results = []
            #         for text in texts:
            #             result = [idx2word[t] for t in text[1:]]
            #             if result not in result_b:
            #                 result_b.append(list(set(result)))
            #
            # new_results = sorted(zip(score_b, result_b),
            #                      key=lambda p: p[0], reverse=True)
            # new_score, new_result = zip(*new_results)
            # # print('new_results: ', new_results)
            # # print('new_result: ', new_result)
            # result_dict["generated_results"][b] = list(new_result)
        return result_dict












    # def inference(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx, kws):
    #     """
    #     :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    #     :param src_lens: a list containing the length of src sequences for each batch, with len=batch
    #     :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    #     :param src_mask: a FloatTensor, [batch, src_seq_len]
    #     :param oov_lists: list of oov words (idx2word) for each batch, len=batch
    #     :param word2idx: a dictionary
    #     """
    #     self.model.eval()
    #     batch_size = src.size(0)
    #     max_kp_num = self.model.decoder.max_kp_num
    #     max_kp_len = self.model.decoder.max_kp_len
    #     vocab_size = self.model.decoder.vocab_size
    #     # print("decoder vocab: ", vocab_size)
    #
    #     # Encoding
    #     memory_bank = self.model.encoder(src, src_lens, src_mask)
    #     # print("memory_bank: ", memory_bank)
    #     state = self.model.decoder.init_state(memory_bank, src_mask)
    #     # print("state: ", state)
    #     control_embed, kws_embed = self.model.decoder.forward_seg(state, kws)
    #     # print("kws_embed: ", kws_embed)
    #     # print("src_oov: ", src_oov)
    #
    #     max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
    #     attn_dict_list = []
    #     decoder_score_list = []
    #     output_tokens = src.new_zeros(batch_size, max_kp_num, max_kp_len + 1)
    #     output_tokens[:, :, 0] = word2idx[io.BOS_WORD]
    #     # print("output_0: ", output_tokens[:, :, 0])
    #     for t in range(1, max_kp_len+1):
    #         decoder_inputs = output_tokens[:, :, :t]
    #         decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(vocab_size - 1), word2idx[io.UNK_WORD])
    #         # if t == 1:
    #         #     print("decode_input: ", decoder_inputs)
    #         decoder_dist, attn_dist = self.model.decoder(decoder_inputs, state, src_oov, max_num_oov, kws_embed, src)
    #         # if t == 1:
    #         #     print("decoder_dist: ", decoder_dist.size())
    #         attn_dict_list.append(attn_dist.reshape(batch_size, max_kp_num, 1, -1))
    #         decoder_score_list.append(decoder_dist.max(-1)[0].reshape(batch_size, max_kp_num, 1))
    #
    #         _, tokens = decoder_dist.max(-1)
    #         output_tokens[:, :, t] = tokens
    #     # print("tokens: ", tokens)
    #     output_tokens = output_tokens[:, :, 1:].reshape(batch_size, max_kp_num*max_kp_len)[:, None]  # [batch_size, 1, max_kp_num, max_kp_len]
    #     attn_dicts = torch.cat(attn_dict_list, -2).reshape(batch_size, max_kp_num*max_kp_len, -1)[:, None]  # [batch_size, 1, max_kp_num, max_kp_len, max_src_len]
    #     decoder_scores = torch.cat(decoder_score_list, -1).reshape(batch_size, max_kp_num * max_kp_len)[:, None]
    #
    #     # Extract sentences
    #     result_dict = {"predictions": [], "attention": [], "decoder_scores": []}
    #     for b in range(batch_size):
    #         result_dict["predictions"].append(output_tokens[b])
    #         result_dict["attention"].append(attn_dicts[b])
    #         result_dict["decoder_scores"].append(decoder_scores[b])
    #     return result_dict
