import os
import csv

def calculate_num_unique_words(folderpath):
    vocabulary = {}
    files = os.listdir(folderpath)
    for file in files:
        file_path = os.path.join(folderpath, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split()
        for word in text:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
    return len(vocabulary)

def calculate_n_most_frequent_words(folderpath, n):
    vocabulary = {}
    files = os.listdir(folderpath)
    for file in files:
        file_path = os.path.join(folderpath, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split()
        for word in text:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

    sorted_vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_vocabulary.items())[:n]

def caclulate_statistics(folderpath):
    pass

def calculate_new_word(folderpath):
    pass

def calculate_new_word_reuse(folderpath):
    pass

def calculate_bigram(folderpath):
    pass

def calculate_bigram_reuse(folderpath):
    pass

def calculate_trigram(folderpath):
    pass

def calculate_trigram_reuse(folderpath):
    pass

def calculate_new_word_combs(folderpath):
    pass

def calculate_backward_cosine(folderpath):
    pass

def calculate_forward_cosine(folderpath):
    pass

if __name__=="__main__":
    out = calculate_n_most_frequent_words("test/", 10)
    print(out)