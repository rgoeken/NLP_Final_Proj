import string

import markovify
import random
import numpy as np
import os
import re
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
# from keras.layers.core import
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from collections import Counter
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate

#first two functions were used to see how unique the outputs were
def count_words_fast(text_file):
    if '.txt' == text_file[len(text_file) - 4:]:
        text = open(text_file, encoding='utf-8').read()
    else:
        text = text_file
    word_counts = Counter()
    skips = [".", ", ", ":", ";", "'", '"']
    for line in text:
        t = line.split(" ")
        for word in t:
            word = word.lower()
            for ch in skips:
                word = word.replace(ch, "")
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    return word_counts

def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)

#utilized for LSTM
def create_network(depth, X, y):
    #took over 10 hours
    '''model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))'''

    # generated the same next word
    model = Sequential()
    model.add(LSTM(4, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    for i in range(depth):
        model.add(LSTM(8, return_sequences=True))
    model.add(LSTM(2, return_sequences=True))
    model.summary()
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.compile(optimizer='rmsprop', loss='mse')
    if saved_weights in os.listdir(".") and not train_mode:
        model.load_weights(str(saved_weights))
        print("loading saved network: " + saved_weights)
    return model

#Lstm generation of text
def generate_text(lines_text, data, model):
    chars = list(set(lines_text.split()))
    words = data.split(" ")
    x_data, y_data = build_dataset(words)
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    generated = data + " "

    for i in range(10):
        predicted = model.predict(x_data, verbose=0)[0]
        print(predicted)
        # converting the vector to an integer
        next_index = np.argmax(predicted)
        print(next_index)
        # converting the integer to a character
        next_char = int_to_char[next_index]
        print(next_char)
        # add the character to results
        generated = generated + " " + next_char
        # shift seed and the predicted character
        data = data[1:] + " " + next_char

        print(generated)
        words = generated.split(" ")
        x_data, y_data = build_dataset(words)


    return generated

#generate the markov chains
def markov(text_file):
    read = open(text_file, "r", encoding='utf-8').read()
    # default is two for the state_size
    read = read.split("\n")
    while "<|endoftext|>" in read:
        read.remove("<|endoftext|>")
    read = "\n".join(read)
    text_model = markovify.Text(read)
    return text_model

#see if this can help with training LSTM
def syllables(line):
    count = 0
    line = line.strip()
    for word in line.split(" "):
        vowels = 'aeiouy'
        word = word.lower().strip(string.punctuation)
        try:
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le'):
                count += 1
            if count == 0:
                count += 1
        except:
            continue
    return count / maxsyllables

#get the training files ready for evaluation
def split_lyrics_file(text_file):
    text = open(text_file, encoding='utf-8').read()
    text = text.split(split_char)
    while "" in text:
        text.remove("")
    while "<|endoftext|>" in text:
        text.remove("<|endoftext|>")
    return text

#with markov help to generate the chains
def generate_lines_text(text_model, lines_text):
    bars = []
    lyriclength = len(lines_text)
    count = 0
    while count < lyriclength / 9 and count < lyriclength * 2:
        bar = text_model.make_sentence(max_overlap_ratio=.49, tries=100)
        if type(bar) != type(None):
            bars.append(bar)
            count += 1
    return bars

#Initial version of creating the dataset for LSTM
def build_dataset(words):
    dataset = []
    c = Counter()
    while '' in words:
        words.remove('')

    while '<|endoftext|>' in words:
        words.remove('<|endoftext|>')
    for word in words:
        if word in c:
            c[word] += 1
        else:
            c[word] = 1
    for word in set(words):
        line_list = [word, c[word] + syllables(word)]
        dataset.append(line_list)
    x_data = []
    y_data = []
    for i in range(len(dataset) - 3):
        line1 = dataset[i][1:]
        line2 = dataset[i + 1][1:]
        line3 = dataset[i + 2][1:]
        line4 = dataset[i + 3][1:]
        x = [line1[0], line2[0]]
        x = np.array(x)
        x = x.reshape(1, 2)
        x_data.append(x)
        y = [line3[0], line4[0]]
        y = np.array(y)
        y = y.reshape(1, 2)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

#not enough space needed ~242 GB to work this way
def build_dataset_2(lines):
    words = " ".join(lines).split(" ")
    unique_words = list(set(words))
    unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
    print(unique_word_index)
    LENGTH_WORD = 100
    next_words = []
    prev_words = []
    for j in range(len(words) - LENGTH_WORD):
        prev_words.append(words[j:j + LENGTH_WORD])
        next_words.append(words[j + LENGTH_WORD])
    print(prev_words[0])
    print(next_words[0])
    X = np.zeros((len(prev_words), LENGTH_WORD, len(unique_words)))
    print(X)
    Y = np.zeros((len(next_words), len(unique_words)))
    print(Y)
    for i, each_words in enumerate(prev_words):
        for j, each_word in enumerate(each_words):
            print(i, j)
            X[i, j, unique_word_index[each_word]] = 1
            Y[i, unique_word_index[next_words[i]]] = 1
    return X, Y

#train the LSTM model
def train(x_data, y_data, model):
    model.fit(np.array(x_data), np.array(y_data),
              batch_size=2,
              epochs=5,
              verbose=1)
    model.save_weights(saved_weights)

#choose which path you want to go down
def main(depth, train_mode, config_path, checkpoint_path, vocab_path, encoder_path, file_to_write):
    endings = "!.?"
    if train_mode:
        lines_text = split_lyrics_file(text_file)
        lines = " ".join(lines_text).split(" ")
        x_data, y_data = build_dataset(lines)
        model = create_network(depth, x_data, y_data)
        train(x_data, y_data, model)
        index = random.randint(0, len(lines_text) - 1)
        output = generate_text(" ".join(lines_text), lines_text[index], model)

        f = open(file_to_write, "w", encoding='utf-8')
        f.write(output)
        f.close()
    if not train_mode:
        text_model = markov(text_file)
        text_input = split_lyrics_file(text_file)
        lines_text = generate_lines_text(text_model, text_input)
        print('Load model from checkpoint...')
        model_q = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        print('Load BPE from files...')
        bpe = get_bpe_from_files(encoder_path, vocab_path)
        print('Generate text...')

        index = random.randint(0, len(lines_text)-1)
        print(lines_text[index])

        index2 = random.randint(0, len(lines_text) - 1)
        print(lines_text[index2])

        input = lines_text[index] + " " + lines_text[index2]
        '''
        output = generate(model_q, bpe, [input], length=200, top_k=3)
        while "<|endoftext|>" in output[0]:
            output[0] = " ".join(output[0].split("<|endoftext|>"))
        print(output[0])

        f = open(file_to_write, "w", encoding='utf-8')
        f.write(output[0])
        f.close()

        '''
        #this allows for cleaner output
        output = generate(model_q, bpe, [input], length=100, top_k=3)
        print(output)
        n = 0
        while n < 30:
            f = open(file_to_write, "a", encoding='utf-8')
            while "<|endoftext|>" in output[0]:
                output[0] = " ".join(output[0].split("<|endoftext|>"))


            to_file = output[0].split("\n")

            if len(to_file) == 1:
                to_file = to_file[0].split(".")
                for i in range(0, len(to_file)-1):
                    to_file[i] += "."
            while "" in to_file:
                to_file.remove("")

            for i in to_file[:-1]:
                if re.search("[A-Za-z]", i):
                    f.write(i)
                    word_counts = count_words_fast(i)
                    (num_unique, counts) = word_stats(word_counts)
                    print(num_unique, sum(counts))
                    f.write("\n")
            if re.search("[A-Za-z]", to_file[-1]) and to_file[-1][-1] in endings:
                index = random.randint(0, len(lines_text) - 1)
                to_find = to_file[-1] + " " + lines_text[index]
                print(to_find)
                output = generate(model_q, bpe, [to_find], length=100, top_k=3)
                print(output)
                word_counts = count_words_fast(output)
                (num_unique, counts) = word_stats(word_counts)
                print(num_unique, sum(counts))
            else:
                print(to_file[-1])
                output = generate(model_q, bpe, [to_file[-1]], length=100, top_k=3)
                print(output)
                word_counts = count_words_fast(output)
                (num_unique, counts) = word_stats(word_counts)
                print(num_unique, sum(counts))
            f.close()
            n += 1

        f = open(file_to_write, "a", encoding='utf-8')
        while "<|endoftext|>" in output[0]:
            output[0] = " ".join(output[0].split("<|endoftext|>"))

        to_file = output[0].split("\n")

        if len(to_file) == 1:
            to_file = to_file[0].split(".")
            for i in range(0, len(to_file) - 1):
                to_file[i] += "."

        for i in to_file[:-1]:
            if re.search("[A-Za-z]", i):
                f.write(i)
                print(i)
                word_counts = count_words_fast(output)
                (num_unique, counts) = word_stats(word_counts)
                print(num_unique, sum(counts))
                f.write("\n")
        if re.search("[A-Za-z]", to_file[-1]) and to_file[-1][-1] in endings:
            f.write(to_file[-1])
            print(to_file[-1])
            word_counts = count_words_fast(output)
            (num_unique, counts) = word_stats(word_counts)
            print(num_unique, sum(counts))
            f.write("\n")
        f.close()

#the LSTM test run with different dataset and training model
def testing(filename, saved_weights):
    # load ascii text and covert to lowercase
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    temp = raw_text.split(" ")
    while '<|endoftext|>' in temp:
        temp.remove('<|endoftext|>')
    raw_text = " ".join(temp)
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    # define the LSTM model
    model = create_network(depth, X, y)
    train(X, y, model)
    # load the network weights
    model.load_weights(saved_weights)

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        print(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

#model_folder = 'C:\\Users\\rache\\gpt-2\\models\\117M' # default model

model_folder = 'C:\\Users\\rache\\git\\gpt-2\\models\\117M' #self trained model?
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')

depth = 4
maxsyllables = 50 #average syllables in a sentence is 29?

#save states
saved_weights = "saved_weights_temp.txt"
story_file_LSTM = "scifi_auto_generated_LSTM.txt"

story_file_markov = "scifi_auto_generated_markov.txt"
split_char = '\n'

#text_file = "scifi_v3.txt" #from kaggle
text_file = "scifi_v3_gpt_2.txt" #created from https://blog.reedsy.com/short-stories/science-fiction/

word_counts = count_words_fast(text_file)
(num_unique, counts) = word_stats(word_counts)
print(num_unique, sum(counts))


#train_mode = True
#main(depth, train_mode, config_path, checkpoint_path, vocab_path, encoder_path, story_file_LSTM)
testing(text_file, saved_weights)
train_mode = False
main(depth, train_mode, config_path, checkpoint_path, vocab_path, encoder_path, story_file_markov)


print(num_unique, sum(counts))
word_counts = count_words_fast(story_file_LSTM)
(num_unique, counts) = word_stats(word_counts)
print(num_unique, sum(counts))

word_counts = count_words_fast(story_file_markov)
(num_unique, counts) = word_stats(word_counts)
print(num_unique, sum(counts))

