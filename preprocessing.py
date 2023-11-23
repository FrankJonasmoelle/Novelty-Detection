import nltk    
import spacy
import re
import os
import concurrent.futures
import pandas as pd

nltk.download("stopwords")
nlp = spacy.load("sv_core_news_sm", disable=['ner', 'parser', 'textcat'])


def remove_whitespace(text):
    return text.strip()

def remove_token_whitespace(ls):
    return [token.strip() for token in ls]

def remove_special_characters(text):
    return re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', text)

def to_lowercase(text):
    return text.lower()

def tokenize(text):
    doc = nlp(text)
    return [item.text for item in doc]

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def remove_stopwords(ls):
    stopwords = nltk.corpus.stopwords.words('swedish') 
    return [item for item in ls if item not in stopwords]

def remove_letters(ls):
    return [item for item in ls if len(item) > 1]

def preprocess(text):
    text = remove_whitespace(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = lemmatize(text)
    text = tokenize(text)
    text = remove_token_whitespace(text)
    text = remove_stopwords(text)
    text = remove_letters(text)
    return text

def preprocess_files(input_folder="data/", output_folder="data_preprocessed/"):
    """takes an input folder with .txt files and applies preprocessing. The resulting preprocessed files are saved 
    in the output_folder"""
    files = os.listdir(input_folder)
    for file in files:
        # preprocess file only if it hasn't been done yet
        output_path = os.path.join(output_folder, file)
        if os.path.exists(output_path):
           print("file already preprocessed")
        else:
            if file.endswith(".txt"):
                file_path = os.path.join(input_folder, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    preprocessed_text = preprocess(text)
                # save preprocessed text to new folder
                with open(output_path, "w", encoding="utf-8") as f:
                    for i, token in enumerate(preprocessed_text):
                        f.write(token)
                        if i % 20 == 0:
                            f.write("\n")
                        else:
                            f.write(" ")

def preprocess_file_helper(file_path, output_path):
    """Helper function to process a single file."""
    if os.path.exists(output_path):
        print(f"File {output_path} already preprocessed")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    preprocessed_text = preprocess(text)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, token in enumerate(preprocessed_text):
            f.write(token)
            if i % 20 == 0:
                f.write("\n")
            else:
                f.write(" ")

def preprocess_files_parallel(input_folder="data/", output_folder="data_preprocessed/"):
    """Parallelized version of preprocess_files."""
    files = os.listdir(input_folder)
    file_paths = [os.path.join(input_folder, file) for file in files]
    output_paths = [os.path.join(output_folder, file) for file in files]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(preprocess_file_helper, file_paths, output_paths)

def get_most_frequent_keywords(input_folder="data_preprocessed/", output_path="frequencies.csv"):
    """returns a list of all words, sorted by their frequency"""
    vocabulary = {}
    files = os.listdir(input_folder)
    for file in files:
        file_path = os.path.join(input_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split()
        for word in text:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
    sorted_vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True))
    
    # save it to xlsx file
    df = pd.DataFrame([sorted_vocabulary])
    df.to_csv(output_path, index=False)

    return list(sorted_vocabulary.items())

if __name__=="__main__":
    #preprocess_files(input_folder="data/", output_folder="test_preprocessed/")
    #ls = get_most_frequent_keywords("data_preprocessed/")
    #print(ls)'
    

    preprocess_files_parallel(input_folder="data/", output_folder="test_preprocessed")
