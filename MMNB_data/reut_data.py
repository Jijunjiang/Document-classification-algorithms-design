
import random

def mergedic(dic2, dic1):
    for (k, v) in dic1.items():
        if dic2.has_key(k):
            dic2[k] += v
        else:
            dic2[k] = v
    return dic2



def generate_train(list_size, size, datapath):


    train_para = []
    train_topic = []
    test_para = []
    test_topic = []
    dic_train = {}
    corp = {}
    sumofwords = {}
    smooth = {}

    paragraph, topic = laod_data(datapath)
    topic_set = list(set(topic))
    print 'total topics:\n'
    print topic_set
    word_size = range(list_size)
    trainID = random.sample(word_size, size)
    testID = [i for i in word_size if i not in trainID]
    for i in trainID:
        train_para.append(paragraph[i])
        train_topic.append(topic[i])

    for i in testID:
        test_para.append(paragraph[i])
        test_topic.append(topic[i])

    for i in trainID:
        for ele in topic_set:
            if train_topic[i] == ele:
                dic_train[ele] = mergedic(dic_train[ele], train_para[i])
    for i in word_size:
        for ele in topic_set:
            if topic[i] == ele:
                corp[ele] = mergedic(corp[ele], paragraph[i])

    for (t, d) in corp.items():
        for (k, v) in d.items():


    for ele in topic_set:








def laod_data(datapath):
    whole_dic = {}
    whole_data = []
    topic_data = []
    with open(datapath) as fileobj:
        data = fileobj.read()
    for line in data:
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
    return whole_data, topic_data


