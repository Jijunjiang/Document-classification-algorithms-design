import random
import sys
import math
import numpy
import os

def mergedic(dic2, dic1):
    for (k, v) in dic1.items():
        if dic2.has_key(k):
            dic2[k] += v
        else:
            dic2[k] = v
    return dic2

def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %s" % (
                            '=' * int(math.floor(cur * 50 / total)),
                            percent))
    sys.stdout.flush()


def generate_train(size, datapath):


    train_para = []
    train_words = {}
    train_topic = []
    test_para = []
    test_topic = []
    dic_train = {}
    corp = {}
    corp_words = {}
    sumofwords = {}
    smooth = {}

    paragraph, topic = laod_data(datapath)
    list_size = len(paragraph)
    topic_set = list(set(topic))
    print 'total topics:'
    print topic_set
    word_size = range(list_size)
    trainID = random.sample(word_size, size)
    testID = [i for i in word_size if i not in trainID]
    for i in trainID:
        train_para.append(paragraph[i])
        train_topic.append(topic[i])
    train_topic_set = list(set(train_topic))
    print 'train topics:'
    print train_topic_set

    for i in testID:
        test_para.append(paragraph[i])
        test_topic.append(topic[i])


    # the training dictionary key is topic, value is paragraph   
    for i in range(len(trainID)):
        for ele in topic_set:
            if train_topic[i] == ele:
                if not dic_train.has_key(ele):
                    dic_train[ele] = {}
                dic_train[ele] = mergedic(dic_train[ele], train_para[i])


    # the corp dictionary key is topic, value is paragraph
    for i in word_size:
        for ele in topic_set:
            if topic[i] == ele:
                if not corp.has_key(ele):
                    corp[ele] = {}
                corp[ele] = mergedic(corp[ele], paragraph[i])


    # corp dictionary of words, key is words, value is number
    # the number of different word in global 
    for (t, p) in corp.items():
        for (k, v) in p.items():
            if corp_words.has_key(k):
                corp_words[k] += v
            else:
                corp_words[k] = v
                

    # for world in training set
    for (t, p) in dic_train.items():
        for (k, v) in p.items():
            if train_words.has_key(k):
                train_words[k] += v
            else:
                train_words[k] = v
                


    # THE NUMBER OF DIFFERENT WORDS IN SEPERATED TOPICS
    # topic_words_count = [dic1, dic2...], dict1 = {word1:number, word2: num..}
    # in the order of train topic set topic_words_count
    
    topic_words_count = []
    for ele in train_topic_set:
        catalogue = {}
        for (w, n) in dic_train[ele].items():
            if catalogue.has_key(w):
                catalogue[w] += n
            else:
                catalogue[w] = n
        topic_words_count.append(catalogue)


    # the word probability in different topics 
    # word_prob_dtopic = [dict1, dict2...], dict1 = {word1:p1...} 
    word_prob_dtopic = []   
    for diction in topic_words_count:
        word_prob= {}
        for (w, n) in diction.items():
            word_prob[w] = float(n)/train_words[w]
        word_prob_dtopic.append(word_prob)

    # words count for different topics
    words_topic_sum = {}
    for i in range(len(train_topic_set)):
        sum = 0
        for (k,v) in topic_words_count[i].items():
            sum += v
        words_topic_sum[train_topic_set[i]] = sum



    #initialize confusion matrix
    confusion_matric = numpy.zeros((len(topic_set), len(topic_set)))

    # laplace smooth value
    T = len(train_para)
    for ele in topic_set:
        if ele in train_topic_set:
            smooth[ele] = 1.0/(T + words_topic_sum[ele])
        else:
            smooth[ele] = 1.0/T

    num = 0
    for ele in test_para:
        mini = -sys.maxint
        macthtopic = ""
        j = 0
        for dicts in word_prob_dtopic:
            logval = 0
            for (k,v) in ele.items():
                for i in range(v):
                    if dicts.has_key(k):
                        logval += math.log(dicts[k])
                    else:
                        logval += math.log(smooth[test_topic[num]])
                if logval > mini:
                    mini = logval
                    macthtopic = train_topic_set[j]

            j += 1
        
        column = topic_set.index(test_topic[num])
        row = topic_set.index(macthtopic)
        confusion_matric[row][column] += 1
        num += 1
        progressbar(float(num), float(len(test_para)))

    print '\nConfusion Matrix(row is target value, colum is predict value): '
    for ele in confusion_matric:
        print ele



def laod_data(datapath):

    whole_dic = {}
    whole_data = []
    topic_data = []
    with open(datapath) as fileobj:
        while 1:
            line = fileobj.readline()
            if not line:
                break
            try:
                article_dic = {}
                [topic, article] = line.split('@')
                words = article.split(',')
                for word in words:
                    ele = word.split(':')
                    article_dic[ele[0]] = (int)(ele[1])
                if topic != '':
                    for ele in topic.split(','):
                        whole_data.append(article_dic)
                        topic_data.append(ele)
            except: ValueError

    return whole_data, topic_data

if __name__ == '__main__':

    size = 5000
    datapath = 'output1.txt'
    generate_train(size, datapath)


