
import pickle
import random
import math
import numpy

dic = {'positive': 1, 'negative': 0}


def mergedic(dic2, dic1):
    for (k, v) in dic1.items():
        if dic2.has_key(k):
            dic2[k] += v
        else:
            dic2[k] = v
    return dic2

def load_arff(infile):
    words = []
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

def generate_SFM(list_size, size, paragraph):

    train_para = []
    test_para = []   #test paragraph(document)
    test_set = {}
    Y_train = []
    dictword = {}
    dictword_pos = {}   #store training data of positive
    dictword_neg = {}


    P_C_W_L_P = {} #possibility of P(C|Wi) in labeled data c = possitive
    P_C_W_L_N = {}
    P_W_L = {} #possibility of P(Wi) in labeled data
    P_W_U = {} #P(Wi) in unlabeled data

    P_W_C_P = {} #possibility of P(Wi|C) c = possitive
    P_W_C_N = {}


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
        else:
            dictword_neg = mergedic(dictword_neg, word_count)

    print 'merge complete'

    test_sum = 0
    for (lable, word_count) in test_para:
        for (k, v) in word_count.items():
            if P_W_U.has_key(k):
                test_set[k] += v
            else:
                test_set[k] = v
            test_sum += v

    # generate P(C,Wi) P(Wi) and P(Wi) of unlabeled--------------------------------------

    for (k,v) in test_set.items():
        P_W_U[k] = float(v) / test_sum


    # generate PWU-----------------------------------------------------------------------
    train_sum = 0
    for (k, v) in dictword.items():
        train_sum += v
    train_sump = 0
    for (k, v) in dictword_pos.items():
        train_sump += v
    for (k,v) in dictword.items():
        if dictword_pos.has_key(k):
            P_C_W_L_P[k] = float(dictword_pos[k]) / v
        if dictword_neg.has_key(k):
            P_C_W_L_N[k] = float(dictword_neg[k]) / v
        P_W_L[k] = float(v)/train_sum
    T = len(dictword) + len(test_set)
    nump = 1.0 / (train_sump + T)
    numn = 1.0 / (train_sum - train_sump + T)

    top_P = {}
    top_N = {}
    bot_sum_P = 0
    bot_sum_N = 0

    for (k, v) in test_set.items():
        if dictword_pos.has_key(k):
            top_P[k] = P_C_W_L_P[k] * P_W_U[k]
        else:
            top_P[k] = 0
        if dictword_neg.has_key(k):
            top_N[k] = P_C_W_L_N[k] * P_W_U[k]
        else:
            top_N[k] = 0
        bot_sum_P += top_P[k]
        bot_sum_N += top_N[k]

    for (k, v) in top_P.items():
        if v == 0:
            P_W_C_P[k] = nump
        else:
            P_W_C_P[k] = top_P[k] / float(bot_sum_P)
    for (k, v) in top_N.items():
        if v == 0:
            P_W_C_N[k] = numn
        else:
            P_W_C_N[k] = top_N[k] / float(bot_sum_N)

    return (P_W_C_P, P_W_C_N, test_para)

def testing(dict_pos, dict_neg, test_para):
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
    # pdb.set_trace()
    for i in range(len(Y_test)):

        sumlogp = 0
        sumlogn = 0
        for item in X_text[i]:

            if dict_pos.has_key(item):
                sumlogp += math.log(dict_pos[item])
            else:
                raise NameError

            if dict_neg.has_key(item):
                sumlogn += math.log(dict_neg[item])
            else:
                raise NameError

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
    # F1P = 2 * precision1 * recall1 / (precision1 + recall1)
    precision2 = float(TN) / (TN + FN)
    recall2 = float(TN) / (TN + FP)
    precision = (precision1 + precision2) / 2
    recall = (recall1 + recall2) / 2
    Macro_ave = 2 * precision * recall / (precision + recall)
    Macro_ave2 = precision1 * recall1 / (precision1 + recall1) + precision2 * recall2 / (precision2 + recall2)
    F1 = float(2 * TP) / (2 * TP + FP + FN)

    TP_ave = float(TP + TN) / 2
    FP_ave = float(FP + FN) / 2
    FN_ave = float(FP + FN) / 2
    K = float(TP_ave) / (TP_ave + FP_ave)
    Micro_ave = K
    print [TP, FP, TN, FN]
    print 'F1 score****: ' + str(F1)
    return (Macro_ave, Macro_ave2, Micro_ave)






#----------------------------------------mian------------------------------------------------
x_record = []


load_arff('/Users/apple/Desktop/reading/machine_495/Zhao_comparison/zhao_data/toys_games.arff')
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
mac2=[]
mic = []
turn = 10
sum = numpy.zeros(4)
#for test_size in [1000, 5000, 10000, 20000, 40000, 70000, 100000, 120000]:
for j in range(turn):
    mac = []
    mac2 = []
    mic = []
    for i in [64, 128, 256, 512]:
        (P_W_C_P, P_W_C_N, test_para) = generate_SFM(list_size,i,paragraph)
        (Macro_ave, Macro_ave2, Micro_ave) = testing(P_W_C_P, P_W_C_N, test_para)
        # with open('training.txt', 'r') as f:
        #     (dict_neg, dict_pos, test_para, nump, numn) = pickle.load(f)

        print 'load complete'

        mac.append(Macro_ave)
        mic.append(Micro_ave)

        mac2.append(Macro_ave2)
    sum += numpy.array(mac)

print sum
print sum/turn
print mac2


