import jieba
import json
import numpy as np
import pickle
import random
import tensorflow as tf
import tflearn

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ["？", "的", "。", "！", "你们", "是", "有", "吗", "我", "想", "我能", "不", "什么", "最"]
suggest_words = ["老板好", "下次见"]
for word in suggest_words:
    jieba.suggest_freq(word, True)

for intent in intents['intents']:
    for phrase in intent['phrases']:
        w = list(jieba.cut(phrase, cut_all=False))
        words.extend(w)
        documents.append((w, intent['intent']))
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

words = [w for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "words", words)

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='saved/tflearn_logs')
model.fit(train_x, train_y, n_epoch=1500, batch_size=5, show_metric=True)
model.save('saved/model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
            open("saved/training_data", "wb"))
