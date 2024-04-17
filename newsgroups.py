from collections import Counter
import torch
import csv
import sys
import math
import random
import re


def remove_punctuation(text):
    cleaned_text = re.sub(r'[^\w\s]|\d', ' ', text)
    return cleaned_text.lower()


def split_and_clean_sentences(data, stopwords):
    final_data = []
    for sentence, _ in data:
        temp = []
        cleaned_sentence = remove_punctuation(sentence)
        words = cleaned_sentence.split()
        for word in words:
            if word not in stopwords:
                temp.append(word)
        final_data.append(temp)
    return final_data


def pad_and_redact(data, labels, word_indices):
    final_data = [[word_indices[word] if word in word_indices else 1 for word in words[:50]] + [0] * max(0, 50 - len(words)) for words in data]
    tensor_final_data = [(torch.tensor(data, dtype=torch.float32), torch.tensor(label)) for data, label in zip(final_data, labels)]
    return tensor_final_data


def newsgroups_featurize(train_data, val_data, dev_data, test_data, feature_type):
    """ Featurizes an input for the sst2 domain.

    Inputs:
        train_data: The training data.
        val_data: The validation data.
        dev_data: The development data.
        test_data: The test data.
        feature_type: Type of feature to be used.
    """
    stopwords = set()
    with open("stopwords.txt", "r", encoding="utf-8") as file:
        for line in file:
            stopwords.add(line.strip())

    if feature_type == 'bow':
        vocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            for word in words:
                if word not in stopwords:
                    vocab.append(word)
        word_counts = Counter(vocab)
        filtered_vocab = [word for word, count in word_counts.items() if count > 3]
        if len(filtered_vocab) <= 1000:
            vocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            vocab = sorted_vocab[:1000]

        def featurize(data):
            mod_input = []
            for sentence, label in data:
                features = {}
                for v in range(len(vocab)):
                    if vocab[v] in sentence:
                        features[v] = 1
                    else:
                        features[v] = 0
                mod_input.append((features, label))
            return mod_input

        X_train = featurize(train_data)
        X_val = featurize(val_data)
        X_dev = featurize(dev_data)
        X_test = featurize(test_data)

    if feature_type == "bi-gram":
        bivocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            n = len(words)
            for i in range(n-1):
                bigram = " ".join(words[i:i+2])
                bivocab.append(bigram)
        word_counts = Counter(bivocab)
        filtered_vocab = [bigram for bigram, count in word_counts.items() if count > 1]
        if len(filtered_vocab) <= 1000:
            bivocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            bivocab = sorted_vocab[:1000]

        univocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            for word in words:
                if word not in stopwords:
                    univocab.append(word)
        word_counts = Counter(univocab)
        filtered_vocab = [word for word, count in word_counts.items() if count > 3]
        if len(filtered_vocab) <= 1000:
            univocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            univocab = sorted_vocab[:1000]
        vocab = bivocab + univocab

        def featurize(data):
            mod_input = []
            for sentence, label in data:
                features = {}
                for v in range(len(vocab)):
                    if vocab[v] in sentence:
                        features[v] = 1
                    else:
                        features[v] = 0
                mod_input.append((features, label))
            return mod_input

        X_train = featurize(train_data)
        X_val = featurize(val_data)
        X_dev = featurize(dev_data)
        X_test = featurize(test_data)

    if feature_type == "tfidf":
        vocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            for word in words:
                if word not in stopwords:
                    vocab.append(word)
        word_counts = Counter(vocab)
        filtered_vocab = [word for word, count in word_counts.items() if count > 3]
        if len(filtered_vocab) <= 1000:
            vocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            vocab = sorted_vocab[:1000]

        train_cleaned = split_and_clean_sentences(train_data, stopwords)
        val_cleaned = split_and_clean_sentences(val_data, stopwords)
        full_train_data = train_cleaned + val_cleaned

        tfidf_scores = {}
        for v in vocab:
            tfidf_scores[v] = 0
        for words in full_train_data:
            temp = set()
            for word in words:
                if word in vocab and word not in temp:
                    tfidf_scores[word] += 1
                    temp.add(word)
        for word, count in tfidf_scores.items():
            tfidf_scores[word] = math.log(len(full_train_data)/float(count + 1))

        tf_idf_vectors = []
        for words in full_train_data:
            tfidf = {}
            word_count = Counter(words)
            for word, count in word_count.items():
                tfidf[word] = count / len(words)
            tf_idf_vectors.append([tfidf.get(word, 0) * tfidf_scores.get(word, 0) for word in vocab])

        def featurize(data):
            mod_input = []
            ind = 0
            for sentence, label in data:
                tfidf_vect = tf_idf_vectors[ind]
                features = {}
                for v in range(len(vocab) + len(tfidf_vect)):
                    if v < len(vocab):
                        if vocab[v] in sentence:
                            features[v] = 1
                        else:
                            features[v] = 0
                    else:
                        features[v] = tfidf_vect[v-len(vocab)]
                mod_input.append((features, label))
                ind += 1
            return mod_input

        X_train = featurize(train_data)
        X_val = featurize(val_data)
        X_dev = featurize(dev_data)
        X_test = featurize(test_data)

    if feature_type == "full":
        bivocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            n = len(words)
            for i in range(n-1):
                bigram = " ".join(words[i:i+2])
                bivocab.append(bigram)
        word_counts = Counter(bivocab)
        filtered_vocab = [bigram for bigram, count in word_counts.items() if count > 1]
        if len(filtered_vocab) <= 1000:
            bivocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            bivocab = sorted_vocab[:1000]

        univocab = []
        for sentence, _ in train_data:
            clean_sentence = remove_punctuation(sentence)
            words = clean_sentence.split()
            for word in words:
                if word not in stopwords:
                    univocab.append(word)
        word_counts = Counter(univocab)
        filtered_vocab = [word for word, count in word_counts.items() if count > 3]
        if len(filtered_vocab) <= 1000:
            univocab = filtered_vocab
        else:
            sorted_vocab = sorted(filtered_vocab, key=lambda word: word_counts[word], reverse=True)
            univocab = sorted_vocab[:1000]
        vocab = bivocab + univocab

        train_cleaned = split_and_clean_sentences(train_data, stopwords)
        val_cleaned = split_and_clean_sentences(val_data, stopwords)
        full_train_data = train_cleaned + val_cleaned

        tfidf_scores = {}
        for v in vocab:
            tfidf_scores[v] = 0
        for words in full_train_data:
            temp = set()
            for word in words:
                if word in vocab and word not in temp:
                    tfidf_scores[word] += 1
                    temp.add(word)
        for word, count in tfidf_scores.items():
            tfidf_scores[word] = math.log(len(full_train_data)/float(count + 1))

        tf_idf_vectors = []
        for words in full_train_data:
            tfidf = {}
            word_count = Counter(words)
            for word, count in word_count.items():
                tfidf[word] = count / len(words)
            tf_idf_vectors.append([tfidf.get(word, 0) * tfidf_scores.get(word, 0) for word in vocab])

        def featurize(data):
            mod_input = []
            ind = 0
            for sentence, label in data:
                tfidf_vect = tf_idf_vectors[ind]
                features = {}
                for v in range(len(vocab) + len(tfidf_vect)):
                    if v < len(vocab):
                        if vocab[v] in sentence:
                            features[v] = 1
                        else:
                            features[v] = 0
                    else:
                        features[v] = tfidf_vect[v-len(vocab)]
                mod_input.append((features, label))
                ind += 1
            return mod_input

        X_train = featurize(train_data)
        X_val = featurize(val_data)
        X_dev = featurize(dev_data)
        X_test = featurize(test_data)

    return X_train, X_val, X_dev, X_test


def newsgroups_data_loader(train_data_filename: str, train_labels_filename: str, dev_data_filename: str, dev_labels_filename: str, test_data_filename: str, feature_type: str, model_type: str):
    """ Loads the data.

    Inputs:
        train_data_filename: The filename of the training data.
        train_labels_filename: The filename of the training labels.
        dev_data_filename: The filename of the development data.
        dev_labels_filename: The filename of the development labels.
        test_data_filename: The filename of the test data.
        feature_type: The type of features to use.
        model_type: The type of model to use.
        IMPORTANT For multilayer, use pytorch to load in

    Returns:
        Training, validation, dev, and test data, all represented as a list of (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    csv.field_size_limit(sys.maxsize)

    def read_csv(data_filepath, labels_filepath):
        with open(data_filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            data = {}
            for row in csv_reader:
                data[row[0]] = row[1]

        if labels_filepath is None:
            return list(zip(data.values(), [-1]*len(data)))

        with open(labels_filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            labels = {}
            for row in csv_reader:
                labels[row[0]] = row[1]

        final_data = []
        for k in data.keys():
            final_data.append((data[k].lower(), labels[k]))
        return final_data

    full_train_data = read_csv(train_data_filename, train_labels_filename)
    dev_data = read_csv(dev_data_filename, dev_labels_filename)
    test_data = read_csv(test_data_filename, None)

    label_to_num = {'alt.atheism': 0,
                    'comp.graphics': 1,
                    'comp.sys.mac.hardware': 2,
                    'comp.sys.ibm.pc.hardware': 3,
                    'comp.os.ms-windows.misc': 4,
                    'comp.windows.x': 5,
                    'misc.forsale': 6,
                    'rec.autos': 7,
                    'rec.motorcycles': 8,
                    'rec.sport.baseball': 9,
                    'rec.sport.hockey': 10,
                    'sci.med': 11,
                    'sci.crypt': 12,
                    'sci.space': 13,
                    'sci.electronics': 14,
                    'soc.religion.christian': 15,
                    'talk.religion.misc': 16,
                    'talk.politics.mideast': 17,
                    'talk.politics.misc': 18,
                    'talk.politics.guns': 19
                    }

    full_train_data = [(d, label_to_num[label]) for d, label in full_train_data]
    dev_data = [(d, label_to_num[label]) for d, label in dev_data]

    random.shuffle(full_train_data)
    val_len = len(full_train_data)//5
    val_data = full_train_data[:val_len]
    train_data = full_train_data[val_len:]
    if model_type == "slp":
        X_train, X_val, X_dev, X_test = newsgroups_featurize(train_data, val_data, dev_data, test_data, feature_type)

    if "mlp" in model_type:
        stopwords = set()
        with open("stopwords.txt", "r", encoding="utf-8") as file:
            for line in file:
                stopwords.add(line.strip())

        train_cleaned = split_and_clean_sentences(train_data, stopwords)
        val_cleaned = split_and_clean_sentences(val_data, stopwords)
        dev_cleaned = split_and_clean_sentences(dev_data, stopwords)
        test_cleaned = split_and_clean_sentences(test_data, stopwords)

        word_indices = {}
        word_indices["PADDING"] = 0
        word_indices["UNKNOWN"] = 0
        index = 2
        for i in range(2):
            data = train_cleaned if i == 0 else val_cleaned
            for sentence in data:
                for word in sentence:
                    if word not in word_indices:
                        word_indices[word] = index
                        index += 1

        train_labels = [x[1] for x in train_data]
        dev_labels = [x[1] for x in dev_data]
        test_labels = [x[1] for x in test_data]
        val_labels = [x[1] for x in val_data]

        X_train = pad_and_redact(train_cleaned, train_labels, word_indices)
        X_dev = pad_and_redact(dev_cleaned, dev_labels, word_indices)
        X_test = pad_and_redact(test_cleaned, test_labels, word_indices)
        X_val = pad_and_redact(val_cleaned, val_labels, word_indices)

    return X_train, X_val, X_dev, X_test
