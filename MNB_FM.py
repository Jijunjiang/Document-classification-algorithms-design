import math, sys
import re
import __builtin__
import pickle
import random
import math
import scipy.optimize
import copy
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
import pdb
words = []
dic = {'positive': 1, 'negative': 0}
ave_pre = []
Macro_ave_2 = []

def mergedic(dic2, dic1):
    for (k, v) in dic1.items():
        if dic2.has_key(k):
            dic2[k] += v
        else:
            dic2[k] = v
    return dic2

def load_arff(infile):

    labels = {}  # {fileID: label}
    labelset = set()
    fileIDs = []  # [fileID]

    MISSLE = {}  # {word: fileID}
    vocabulary = set()  # (word)
    wordcounts = {}  # {fileID: {word: count}}


    f = open(infile)

    # Read word map (ATTRIBUTE section)
    vocab_map = {}
    class_label_id = None

    next_id = 0
    for line in f:
        parts = line.split()

        if len(parts) > 0 and parts[0] == '@ATTRIBUTE':

            word = parts[1]

            if word == 'CLASS_LABEL':  # Ignore CLASS_LABEL as an attribute field
                class_label_id = next_id
            else:
                vocab_map[next_id] = word

            next_id += 1



        elif line == '@DATA\n':
            break
            # Check that ATTRIBUTE section was formatted as expected
    if class_label_id is None:
        raise ValueError('Error: Unexpected arff format. No id for class label was found.')
    if len(vocab_map.keys()) == 0:
        raise ValueError('Error: Unexpected arff format. No vocabulary ATTRIBUTES found.')

        # Read files line by line


    for line in f:

        wordcounts = {}


        # Expecting format {word_id count, word_id count, ...} where at least
        #   one word_id must be class_label_id
        if len(line) < 3 or line[0] != '{' or line[-2] != '}':
            continue

        parts = line[1:-2].split(",")

        label = None
        for part in parts:
            (word_id, value) = part.split(" ")
            word_id = int(word_id)

            # if the paragraph is negtive or positive
            if word_id == class_label_id:
                # Class label for this document
                if label is not None:
                    raise ValueError(
                        'Error: Unexpected arff format. Multiple class labels found for document %d.' % fileID)

                label = value

            else:
                # General word count

                count = int(value)
                              ## filter, only take account the words in more than 3 times
                if word_id in wordcounts:
                    wordcounts[word_id] += count
                else: wordcounts[word_id] = count


        words.append((label, wordcounts))   # for example (positive, dictionary)
    f.close()
    with open('data.txt', 'w') as file_dump:
        pickle.dump((vocab_map, words), file_dump)
    file_dump.close()



def generate_train(list_size, size, paragraph):

    Y = []
    train_para = []
    test_para = []
    Y_train = []

    dictword = {}

    dictword_pos = {}
    dictword_neg = {}

    corp = {}
    corp_p = {}
    corp_n = {}


    word_size = range(list_size)
    trainID = random.sample(word_size, size)
    testID = [i for i in word_size if i not in trainID]
    for i in trainID:
        train_para.append(paragraph[i])

    for i in testID:
        test_para.append(paragraph[i])

    for (lable, word_count) in train_para:
        Y_train.append(dic[lable])
        dictword = mergedic(dictword, word_count)
        if dic[lable] == 1:
            dictword_pos = mergedic(dictword_pos, word_count)
        else :
            dictword_neg = mergedic(dictword_neg, word_count)


    for (lable, word_count) in paragraph:
        Y.append(dic[lable])
        corp = mergedic(corp_p, word_count)
        if dic[lable] == 1:
            corp_p = mergedic(corp_p, word_count)
        else :
            corp_n = mergedic(corp_n, word_count)


    print 'merge complete'
    return (dictword, dictword_pos, dictword_neg, test_para, corp, corp_p, corp_n)

def training(filter, filternum, dictword, dictword_pos, dictword_neg, test_para, corp, corp_p, corp_n):
    wordsum = 0
    word_pos = 0
    corp_sum = 0
    corp_psum = 0
    corp_nsum = 0

    for (k, v) in corp.items():
        corp_sum += v

    for (k, v) in corp_p.items():
        corp_psum += v

    for (k, v) in corp_n.items():
        corp_nsum += v


    dictword_pos_prob = {}
    dictword_neg_prob = {}
    ## no filter
    T = len(corp)
    if filter == False:

        for (k, v) in dictword.items():
            wordsum += v
        print 'word count in trainning dataset:' + str(wordsum)
        for (k, v) in dictword_pos.items():
            word_pos += v

        print 'positive word count in trainning dataset:' + str(word_pos) + ' %% ' + str(float(word_pos)/wordsum)
        print 'negtive word count in trainning dataset:' + str(wordsum - word_pos) + ' %% ' + str(1.0 - float(word_pos)/wordsum)

        for (k, v) in dictword_pos.items():
            dictword_pos_prob[k] = (float(v) + 1) / (word_pos + T)

        for (k, v) in dictword_neg.items():
            dictword_neg_prob[k] = (float(v)+1) / (wordsum - word_pos + T)


        nump = 1.0/(word_pos + T)
        numn = 1.0/(wordsum - word_pos + T)

    else:
        num_drop = 0
        for (k, v) in dictword.items():
            if dictword[k] > filternum:
                wordsum += v
            else:
                num_drop += 1

        print 'word count in trainning dataset:' + str(wordsum)
        for (k, v) in dictword_pos.items():
            if dictword[k] > filternum:
                word_pos += v

        print 'positive word count in trainning dataset:' + str(word_pos) + ' %% ' + str(float(word_pos)/wordsum)
        print 'negtive word count in trainning dataset:' + str(wordsum - word_pos) + ' %% ' + str(1.0 - float(word_pos)/wordsum)

        for (k, v) in dictword_pos.items():
            if dictword[k] > filternum:
                dictword_pos_prob[k] = (float(v) + 1) / (word_pos + T)
            else:
                dictword_pos_prob[k] = 1.0/(word_pos + T)

        for (k, v) in dictword_neg.items():
            if dictword[k] > filternum:
                dictword_neg_prob[k] = (float(v)+1) / (wordsum - word_pos + T)
            else:
                dictword_neg_prob[k] = 1.0 / (wordsum - word_pos + T)


        nump = 1.0/(word_pos + T)
        numn = 1.0/(wordsum - word_pos + T)

        word_pos_num = word_pos
        word_neg_num = wordsum - word_pos
        norm_sum_p = 0
        norm_sum_n = 0
        for (lable, word_count) in test_para:
            for (k, v) in word_count.items():

                # if dictword_pos.has_key(k) == False and dictword_neg.has_key(k) == False:
                #     dictword_pos_prob[k] = nump
                #     dictword_neg_prob[k] = numn
                # else:
                if True:
                    if dictword_pos.has_key(k) == False:
                        value_pos = 0
                    else:
                        value_pos = dictword_pos[k]
                    if dictword_neg.has_key(k) == False:
                        value_neg = 0
                    else:
                        value_neg = dictword_neg[k]
                    N_w_p = max(0.5, value_pos)  # number of word w which is positive
                    N_w_n = max(0.5, value_neg)
                    N_notw_p = word_pos_num - N_w_p
                    N_notw_n = word_neg_num - N_w_n
                    L = float(corp_psum) / corp_nsum
                    K = corp[k] / float(corp_sum) /(float(corp_nsum) / corp_sum)


                    def f(x):
                        return N_w_p/x + N_notw_p/(x-1) + N_w_n * L / (L * x - K) + N_notw_n * L / (L * x - K + 1)

                    try:
                        x = scipy.optimize.newton(f, K / (2 * L))
                    except RuntimeError:
                        x = -1

                    try:
                        if x < 0 or x > 1 or K - L * x < 0 or K - L * x > 1:
                            x = scipy.optimize.newton(f, K / (1.1 * L))

                        if x < 0 or x > 1 or K - L * x < 0 or K - L * x > 1:
                            x = scipy.optimize.newton(f, K / (1.5 * L))
                        if x < 0 or x > 1 or K - L * x < 0 or K - L * x > 1:
                            x = scipy.optimize.newton(f, K / (3 * L))
                        if x < 0 or x > 1 or K - L * x < 0 or K - L * x > 1:
                            x = scipy.optimize.newton(f, K / (10 * L))
                    except RuntimeError:
                        x = -1

                    if x < 0 or x > 1 or K - L * x < 0 or K - L * x > 1:
                        if dictword_pos.has_key(k) == False:
                            dictword_pos_prob[k] = nump
                        if dictword_neg.has_key(k) == False:
                            dictword_neg_prob[k] = numn
                    else:
                        dictword_pos_prob[k] = x
                        dictword_neg_prob[k] = K - L * x



        for (k, v) in dictword_pos_prob.items():
            norm_sum_p += v
        for (k, v) in dictword_neg_prob.items():
            norm_sum_n += v

        for (k, v) in dictword_pos_prob.items():
            dictword_pos_prob[k] = (dictword_pos_prob[k]) / (norm_sum_p)

        for (k, v) in dictword_neg_prob.items():
            dictword_neg_prob[k] = (dictword_neg_prob[k])/(norm_sum_n)


    return (dictword_neg_prob, dictword_pos_prob, test_para, nump, numn)

    # with open('training.txt', 'w') as file_dump:
    #     pickle.dump((dictword_neg, dictword_pos, test_para, nump, numn), file_dump)
    # file_dump.close()





def testing(dict_neg, dict_pos, test_para, nump, numn):

    count = 0
    result = []
    Y_test = []
    X_text = []
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for (lable, word_count) in test_para:
        line = []
        Y_test.append(dic[lable])
        for (k, v) in word_count.items():
            for i in range(v):
                line.append(k)
        X_text.append(line)
    j = 0
    #pdb.set_trace()
    for i in range(len(Y_test)):

        sumlogp = 0
        sumlogn = 0
        for item in X_text[i]:

            if dict_pos.has_key(item):
                sumlogp += math.log(dict_pos[item])
            else:
                sumlogp += math.log(nump)
                #raise NameError

            if dict_neg.has_key(item):
                sumlogn += math.log(dict_neg[item])
            else:
                sumlogn += math.log(numn)
                #raise NameError

        if sumlogp > sumlogn:
            result.append(1)
        else:
            result.append(0)
        if result[j] == 1 and Y_test[j] == 1:
            TP += 1
            count += 1
        elif result[j] == 1 and Y_test[j] == 0:
            FP += 1
        elif result[j] == 0 and Y_test[j] == 0:
            TN += 1
            count += 1
        elif result[j] == 0 and Y_test[j] == 1:
            FN += 1
        # print 'count%%%%:' + str(count)
        # print 'accuracy%%%%%: ' + str(float(count) / (j + 1))
        j += 1

    precision1 = float(TP) / (TP + FP)
    recall1 = float(TP) / (TP + FN)
    #F1P = 2 * precision1 * recall1 / (precision1 + recall1)
    precision2 = float(TN) / (TN + FN)
    recall2 = float(TN) / (TN + FP)
    precision = (precision1 + precision2) / 2
    recall = (recall1 + recall2) / 2
    Macro_ave = 2*precision * recall / (precision + recall)
    Macro_ave2 = precision1 * recall1 / (precision1 + recall1) + precision2 * recall2 / (precision2 + recall2)
    Macro_ave_2.append(Macro_ave2)
    ave_pre.append(precision)
    F1 = float(2*TP)/(2*TP + FP + FN)

    TP_ave = float(TP + TN)/2
    FP_ave = float(FP + FN)/2
    FN_ave = float(FP + FN)/2
    K = float(TP_ave)/(TP_ave + FP_ave)
    Micro_ave = K
    print [TP, FP, TN, FN]
    print 'F1 score****: ' + str(F1)
    return Macro_ave, Micro_ave




x_record = []


load_arff('/Users/apple/Desktop/reading/machine_495/Zhao_comparison/zhao_data/dvd_lucas.arff')
with open('data.txt', 'r') as f:
    data = pickle.load(f)
print 'load complete'
vocabulary_dic = data[0]
paragraph = data[1]
list_size = len(paragraph)
print list_size
T = len(vocabulary_dic)
train_size = 1000
mac = []
mic = []
mac2 = []
mic2 = []
#for test_size in [1000, 5000, 10000, 20000, 40000, 70000, 100000, 120000]:
for i in [64, 128, 256, 512]:
    (dictword, dictword_pos, dictword_neg, test_para, corp, corp_p, corp_n) = generate_train(list_size,i,paragraph)
    (dict_neg, dict_pos, test_para, nump, numn) = training(True,0,dictword, dictword_pos, dictword_neg, test_para, corp, corp_p, corp_n)
    # with open('training.txt', 'r') as f:
    #     (dict_neg, dict_pos, test_para, nump, numn) = pickle.load(f)

    print 'load complete'
    Macro_ave, Micro_ave = testing(dict_neg, dict_pos, test_para, nump, numn)
    mac.append(Macro_ave)
    mic.append(Micro_ave)
    (dict_neg, dict_pos, test_para, nump, numn) = training(False, 0, dictword, dictword_pos, dictword_neg, test_para,corp, corp_p, corp_n)
    Macro_ave, Micro_ave = testing(dict_neg, dict_pos, test_para, nump, numn)

    mac2.append(Macro_ave)
    mic2.append(Micro_ave)
print mac
print mic
print mac2
print mic2
























