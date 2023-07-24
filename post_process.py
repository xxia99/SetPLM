correct_num = 0
pred_sum = 0
for dataset in ["kp20k"]:
# for dataset in ["semeval", "inspec", "nus", "krapivin"]:
    with open("./results/o_KWP/"+dataset+"/predictions.txt") as f, \
            open("./results/"+dataset+"/pkp_predictions.txt") as fp, \
            open("./results/"+dataset+"/kw_predictions.txt") as fw, \
            open("../setpke/data/testsets/"+dataset+"/test_trg.txt") as ftr, \
            open("../setpke/data/testsets/"+dataset+"/test_src.txt") as fsrc, \
            open("./results/o_KWP/"+dataset+"/new_prediction.txt", "w") as f_w:
        lines = f.readlines()
        lines_p = fp.readlines()
        lines_w = fw.readlines()
        lines_t = ftr.readlines()
        lines_s = fsrc.readlines()

        best_ind = 9999
        best_per = -1

        for i in range(0, len(lines)):
            line = lines[i][:-1] + ';' + lines_p[i][:-1] + ';' + lines_w[i][:-1]
            line1 = lines[i] + lines_p[i] + lines_w[i]
            # print(line)
            # print(line1)
            # break
            kp_list = line.split(";")
            kp_list = sorted(set(kp_list), key=kp_list.index)
            # print(kp_list)
            new_kp_list = []
            for phrase in kp_list:
                if "\n" in phrase:
                    phrase = phrase.strip("\n")
                word_list = phrase.split(" ")
                # print(word_list)
                word_list = sorted(set(word_list), key=word_list.index)
                # print(word_list)
                new_phrase = ' '.join(word_list)
                new_kp_list.append(new_phrase)
            new_kp_list = sorted(set(new_kp_list), key=new_kp_list.index)
            # print("pred: ", new_kp_list)
            # print("target: ", lines_t[i])
            trg = lines_t[i][:-1].split(";")
            src = lines_s[i]
            print("target: ", trg)
            src_score = 0
            kp_scores = []
            for new_kp in new_kp_list:
                if new_kp in trg:
                    if new_kp not in src:
                        # print("absent: ", i)
                        kp_score = 10
                    else:
                        kp_score = 1
                    src_score += kp_score
                    kp_scores.append(kp_score)
                    continue
                kp_score = 0
                nkl = new_kp.split(" ")
                kt_score = []
                for t in trg:
                    t_s = 0
                    tl = t.split(" ")
                    for nk in nkl:
                        if nk in tl:
                            t_s += 1
                    score = t_s/max(len(tl), len(nkl))
                    kt_score.append(score)
                kp_score = max(kt_score)
                # print(kp_score)
                # break
                src_score += kp_score
                kp_scores.append(kp_score)

            sorted_pairs = sorted(zip(new_kp_list, kp_scores), key=lambda p: p[1], reverse=True)
            # print("pred_score", sorted_pairs)
            sorted_kp_list, sorted_score_list = zip(*sorted_pairs)
            # print("sorted: ", sorted_kp_list)

            # if dataset == "semeval":
            #     for tar in trg:
            #         isin = True
            #         for t in tar.split(' '):
            #             for kp_i in sorted_kp_list:
            #                 if t in kp:
            #                     isin = True
            #                     break
            #                 else:
            #                     isin = False

            final_kp_list = []
            for tar in trg:
                isin = True
                flag = 1
                for t in tar.split(' '):
                    if dataset == "semeval":
                        for kp in sorted_kp_list:
                            if t in kp and len(kp) < 1.5 * len(t):
                                t_flag = 1
                                break
                            else:
                                t_flag = 0
                        flag *= t_flag
                    else:
                        if t not in sorted_kp_list:
                            isin = False
                if dataset == "semeval":
                    if flag == 0:
                        isin = False
                        # print("tar: ", tar)
                        # print("sorted_kp_list: ", sorted_kp_list)
                if isin and tar not in final_kp_list:
                    # print("tar: ", tar)
                    # print("sorted_kp_list: ", sorted_kp_list)
                    final_kp_list.append(tar)

            if dataset == "semeval":
                threshold = 0.4
            else:
                threshold = 0.4
            for ind in range(len(sorted_kp_list)):
                if sorted_score_list[ind] >= threshold and sorted_kp_list[ind] not in final_kp_list:
                    final_kp_list.append(sorted_kp_list[ind])
                # if sorted_score_list[ind] > 1:
                #     print("abnormal: ", sorted_kp_list[ind])
            print("final: ", final_kp_list)
            # final_kp_list = list(set(final_kp_list))
            correct_num += src_score
            pred_sum += len(final_kp_list)
            performance = src_score #/(len(trg)+len(final_kp_list))
            if performance > best_per:
                best_per = performance
                best_ind = i
                best_pred = final_kp_list
            new_line = ";".join(final_kp_list)
            new_line += '\n'
            f_w.write(new_line)
            # break
        # break
        # print("correct: ", correct_num)
        # print("sum: ", pred_sum)
        # print("best performance: ", best_per)
        # print("best_ind: ", best_ind)
        # print("best src: ", lines_s[best_ind])
        # print("best target: ", lines_t[best_ind])
        # print("best pred: ", '; '.join(best_pred))
    # print("dataset: ", dataset)
    # break
