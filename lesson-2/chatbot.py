# restore all of our data structures
import jieba
import json
import numpy as np
import pickle
import random
import tflearn
import tensorflow as tf

data = pickle.load(open("saved/training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='saved/tflearn_logs')
model.load('saved/model.tflearn')


def bow(sentence, words, show_details=False):
    sentence_words = list(jieba.cut(sentence, cut_all=False))
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# bow("你们营业时间是几点？", words)

ERROR_THRESHOLD = 0.01

context = {"context_set": ""}


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def reply(sentence):
    results = classify(sentence)
    while results:
        for i in intents['intents']:
            if i['intent'] == results[0][0]:
                if "context_set" in i:
                    context["context_set"] = i["context_set"]
                if "context_filter" not in i or \
                        ("context_filter" in i and i["context_filter"] == context["context_set"]):
                    return print(random.choice(i['replies']))

        results.pop(0)
