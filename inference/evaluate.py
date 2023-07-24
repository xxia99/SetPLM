import logging
import os
import time
import torch
import pykp.utils.io as io
from pykp.utils.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.string_helper import *
from utils.functions import time_since
from pykp.utils.label_assign import hungarian_assign


def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_indices, kws_str, kws = batch

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            n_batch += batch_size
            word2idx = opt.vocab['word2idx']
            target = trg_oov if opt.copy_attention else trg

            start_time = time.time()
            if opt.fix_kp_num_len:
                #print("src: ", src)
                #print("shape: ", src.size())
                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                control_embed, kws_embed = model.decoder.forward_seg(state, kws)

                y_t_init = target.new_ones(batch_size, opt.max_kp_num, 1) * word2idx[io.BOS_WORD]
                if opt.set_loss:  # reassign target
                    input_tokens = src.new_zeros(batch_size, opt.max_kp_num, opt.assign_steps + 1)
                    decoder_dists = []
                    input_tokens[:, :, 0] = word2idx[io.BOS_WORD]
                    for t in range(1, opt.assign_steps + 1):
                        decoder_inputs = input_tokens[:, :, :t]
                        decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(opt.vocab_size - 1),
                                                                    word2idx[io.UNK_WORD])

                        decoder_dist, _ = model.decoder(decoder_inputs, state, src_oov, max_num_oov, kws_embed, src, src_mask)
                        input_tokens[:, :, t] = decoder_dist.argmax(-1)
                        decoder_dists.append(decoder_dist.reshape(batch_size, opt.max_kp_num, 1, -1))
                    decoder_dists = torch.cat(decoder_dists, -2)

                    if opt.seperate_pre_ab:
                        mid_idx = opt.max_kp_num // 2
                        # mid_idx = 0
                        pre_reorder_index = hungarian_assign(decoder_dists[:, :mid_idx],
                                                             target[:, :mid_idx, :opt.assign_steps],
                                                             ignore_indices=[word2idx[io.NULL_WORD],
                                                                             word2idx[io.PAD_WORD]])
                        target[:, :mid_idx] = target[:, :mid_idx][pre_reorder_index]
                        trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                        ab_reorder_index = hungarian_assign(decoder_dists[:, mid_idx:],
                                                            target[:, mid_idx:, :opt.assign_steps],
                                                            ignore_indices=[word2idx[io.NULL_WORD],
                                                                            word2idx[io.PAD_WORD]])
                        target[:, mid_idx:] = target[:, mid_idx:][ab_reorder_index]
                        trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]
                    else:
                        reorder_index = hungarian_assign(decoder_dists, target[:, :, :opt.assign_steps],
                                                         [word2idx[io.NULL_WORD],
                                                          word2idx[io.PAD_WORD]])
                        target = target[reorder_index]
                        trg_mask = trg_mask[reorder_index]

                state = model.decoder.init_state(memory_bank, src_mask)  # refresh the state
                input_tgt = torch.cat([y_t_init, target[:, :, :-1]], dim=-1)
                input_tgt = input_tgt.masked_fill(input_tgt.gt(opt.vocab_size - 1), word2idx[io.UNK_WORD])
                decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov, kws_embed, src, src_mask)

            else:
                y_t_init = trg.new_ones(batch_size, 1) * word2idx[io.BOS_WORD]  # [batch_size, 1]
                input_tgt = torch.cat([y_t_init, trg[:, :-1]], dim=-1)
                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.fix_kp_num_len:
                if opt.seperate_pre_ab:
                    mid_idx = opt.max_kp_num // 2
                    # mid_idx = 0
                    pre_loss = masked_cross_entropy(
                        decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, :mid_idx] \
                            .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                        target[:, :mid_idx].reshape(batch_size, -1),
                        trg_mask[:, :mid_idx].reshape(batch_size, -1),
                        loss_scales=[opt.loss_scale_pre],
                        scale_indices=[word2idx[io.NULL_WORD]])
                    ab_loss = masked_cross_entropy(
                        decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, mid_idx:]
                            .reshape(batch_size, opt.max_kp_len * (opt.max_kp_num - mid_idx), -1),
                        target[:, mid_idx:].reshape(batch_size, -1),
                        trg_mask[:, mid_idx:].reshape(batch_size, -1),
                        loss_scales=[opt.loss_scale_ab],
                        scale_indices=[word2idx[io.NULL_WORD]])
                    loss = 1 * pre_loss + 1 * ab_loss
                else:
                    loss = masked_cross_entropy(decoder_dist, target.reshape(batch_size, -1),
                                                trg_mask.reshape(batch_size, -1),
                                                loss_scales=[opt.loss_scale], scale_indices=[word2idx[io.NULL_WORD]])
            else:
                loss = masked_cross_entropy(decoder_dist, target, trg_mask)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += trg_mask.sum().item()

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def evaluate_greedy_generator(data_loader, generator, opt):
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    interval = 1000
    with torch.no_grad():
        word2idx = opt.vocab['word2idx']
        # print(len(word2idx.items()))
        # word2idx = generator.model.encoder.tk.vocab
        idx2word = opt.vocab['idx2word']
        # idx2word = generator.model.encoder.tk.ids_to_tokens
        start_time = time.time()
        for batch_i, batch in enumerate(data_loader):
            # print("batch: ", batch_i)
            if (batch_i + 1) % interval == 0:
                logging.info("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                start_time = time.time()

            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_idx_list, kws_str, kws = batch
            # print("src_str_list: ", src_str_list[0])
            # print('kws_str: ', kws_str[0])
            # break

            if opt.fix_kp_num_len:
                # print(len(word2idx.items()))
                # print(src[0])
                # print(word2idx[io.EOS_WORD])
                # print("src_str_list: ", src_str_list[0])
                # print('kws_str: ', kws_str[0])
                n_best_result = generator.inference(src, src_lens, src_oov, src_mask, oov_lists, word2idx, idx2word, kws, src_str_list)
                # print(len(n_best_result["predictions"]))
                # print(n_best_result["predictions"][0])
                # break
                # print("idx2word: ", len(idx2word.items()))
                pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                     eos_idx=-1,  # to keep all the keyphrases rather than only the first one
                                                     unk_idx=word2idx[io.UNK_WORD],
                                                     replace_unk=opt.replace_unk,
                                                     src_str_list=src_str_list,
                                                     src_mask=src_mask)

                # recover the original order in the dataset
                seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists,
                                       n_best_result['decoder_scores'], kws_str),
                                   key=lambda p: p[0])
                original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists, decoder_scores, kws_str = zip(*seq_pairs)

                # seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, oov_lists,
                #                        kws_str, n_best_result['generated_results']),
                #                    key=lambda p: p[0])
                #
                # original_idx_list, src_str_list, trg_str_2dlist, oov_lists, kws_str, results = zip(*seq_pairs)


                # Process every src in the batch
                for src_str, trg_str_list, pred, oov, decoder_score, kw_str, original_idx in zip(src_str_list, trg_str_2dlist, pred_list,
                                                                           oov_lists, decoder_scores, kws_str, original_idx_list):

                    all_keyphrase_list = split_word_list_from_set(pred[-1], decoder_score[-1].cpu().numpy(),
                                                                  opt.max_kp_len,
                                                                  opt.max_kp_num, io.EOS_WORD, io.NULL_WORD)
                    new_kp_list = []
                    if len(all_keyphrase_list):
                        for i in range(len(all_keyphrase_list)):
                            new_kp = list(set(all_keyphrase_list[i]))
                            # if new_kp not in new_kp_list:
                            new_kp_list.append(new_kp)
                        all_keyphrase_list = tuple(new_kp_list)
                        # all_keyphrase_list += tuple(result)
                        # all_keyphrase_list = tuple(all_keyphrase_list)
                        # print("all_keyphrase_list: ", all_keyphrase_list)
                        # print("\n")


                # for src_str, trg_str_list, oov, kw_str, result, original_idx in zip(
                #         src_str_list, trg_str_2dlist,
                #         oov_lists, kws_str, results, original_idx_list):
                #     all_keyphrase_list = tuple(result)
                    write_example_kp(pred_output_file, all_keyphrase_list)
                # break
            else:
                n_best_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, word2idx)
                pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                     word2idx[io.EOS_WORD],
                                                     word2idx[io.UNK_WORD],
                                                     opt.replace_unk, src_str_list)

                # recover the original order in the dataset
                seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                                   key=lambda p: p[0])
                original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

                # Process every src in the batch
                for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                    # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                    # pred_seq_list: a list of sequence objects, sorted by scores
                    # oov: a list of oov words
                    # all_keyphrase_list: a list of word list contains all the keyphrases \
                    # in the top max_n sequences decoded by beam search
                    all_keyphrase_list = []
                    for word_list in pred:
                        all_keyphrase_list += split_word_list_by_delimiter(word_list, io.SEP_WORD)

                    # output the predicted keyphrases to a file
                    write_example_kp(pred_output_file, all_keyphrase_list)

    pred_output_file.close()

# def evaluate_greedy_generator(data_loader, generator, opt):
#     pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
#     interval = 1000
#     with torch.no_grad():
#         word2idx = opt.vocab['word2idx']
#         # print(len(word2idx.items()))
#         # word2idx = generator.model.encoder.tk.vocab
#         idx2word = opt.vocab['idx2word']
#         # idx2word = generator.model.encoder.tk.ids_to_tokens
#         start_time = time.time()
#         for batch_i, batch in enumerate(data_loader):
#             print("batch: ", batch_i)
#             if (batch_i + 1) % interval == 0:
#                 logging.info("Batch %d: Time for running beam search on %d batches : %.1f" % (
#                     batch_i + 1, interval, time_since(start_time)))
#                 start_time = time.time()
#
#             src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
#             trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_idx_list, kws_str, kws = batch
#
#             if opt.fix_kp_num_len:
#                 # print(len(word2idx.items()))
#                 # print(src[0])
#                 # print(word2idx[io.EOS_WORD])
#                 n_best_result = generator.inference(src, src_lens, src_oov, src_mask, oov_lists, word2idx, kws)
#                 # print(len(n_best_result["predictions"]))
#                 # print(n_best_result["predictions"][0].size())
#                 # print(n_best_result['attention'][0].size())
#                 # break
#                 # print("idx2word: ", len(idx2word.items()))
#                 pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists, src_mask,
#                                                      eos_idx=-1,  # to keep all the keyphrases rather than only the first one
#                                                      unk_idx=word2idx[io.UNK_WORD],
#                                                      replace_unk=opt.replace_unk,
#                                                      src_str_list=src_str_list)
#                 # print("src_str_list: ", src_str_list[0])
#                 # print("trg_str_list: ", trg_str_2dlist)
#                 # print("pred_list: ", pred_list[0])
#
#                 # recover the original order in the dataset
#                 seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists,
#                                        n_best_result['decoder_scores']),
#                                    key=lambda p: p[0])
#                 # print("seq_pairs: ", seq_pairs)
#                 original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists, decoder_scores = zip(*seq_pairs)
#                 # print("pred_list1: ", pred_list)
#
#                 # Process every src in the batch
#                 for src_str, trg_str_list, pred, oov, decoder_score in zip(src_str_list, trg_str_2dlist, pred_list,
#                                                                            oov_lists, decoder_scores):
#                     all_keyphrase_list = split_word_list_from_set(pred[-1], decoder_score[-1].cpu().numpy(),
#                                                                   opt.max_kp_len,
#                                                                   opt.max_kp_num, io.EOS_WORD, io.NULL_WORD)
#
#                     # output the predicted keyphrases to a file
#                     # print("all_keyphrase_list: ", all_keyphrase_list)
#                     write_example_kp(pred_output_file, all_keyphrase_list)
#                 # break
#
#     pred_output_file.close()


def write_example_kp(out_file, kp_list):
    pred_print_out = ''
    # print("kp_list: ", kp_list)
    for word_list_i, word_list in enumerate(kp_list):
        if word_list_i < len(kp_list) - 1:
            pred_print_out += '%s;' % ' '.join(word_list)
        else:
            pred_print_out += '%s' % ' '.join(word_list)
    pred_print_out += '\n'
    out_file.write(pred_print_out)


def preprocess_n_best_result(n_best_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                             src_str_list, src_mask=None):
    predictions = n_best_result['predictions']
    attention = n_best_result['attention']
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, attn_n_best, oov, src_word_list, mask in zip(predictions, attention, oov_lists, src_str_list, src_mask):
        sentences_n_best = []
        # print("pred_n: ", pred_n_best.size())
        # print("attn_n: ", attn_n_best.size())
        # print("mask: ", mask)
        for pred, attn in zip(pred_n_best, attn_n_best):
            # print("pred: ", pred.size())
            # print("attn: ", attn[0])
            attn = torch.multiply(attn, mask)
            # print("attn: ", attn[0])
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            sentences_n_best.append(sentence)
        # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_list.append(sentences_n_best)
    return pred_list
