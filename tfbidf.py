import math
import os
import json
from datetime import datetime
import numpy as np
from numpy.linalg import norm
from concurrent.futures import ThreadPoolExecutor


def generate_json(textfolder_path, jsonfolder_path):
    """generates mapping between the patent id and additional data
    {
        "1":{
            "filepath": "data/1.txt",
            "date": datetime.datetime(1900, 3, 17, 0, 0)
        },
        "2":{
            "filepath": "data/2.txt",
            "date": datetime.datetime(1901, 3, 17, 0, 0)
        },
        "3":{
            "filepath": "data/3.txt",
            "date": datetime.datetime(1902, 3, 17, 0, 0)
        }
    }
    """
    dictionary = {}
    for textfile in os.listdir(textfolder_path):
        id = textfile.split(".")[0] # gets first part of "8.txt"
        text_filepath = os.path.join(textfolder_path, textfile)
        json_filepath = os.path.join(jsonfolder_path, f"SHP_{id}.json")
        try:
            with open(json_filepath, "r") as file:
                json_file = json.load(file)
        except Exception as e:
            print(e)
            continue
        date = json_file["publication_date"]
        if date is None:
            continue # skip patent
        date = datetime.strptime(date, '%Y-%m-%d') # convert to datetime object
        data = {"filepath": text_filepath,
                "date": date}
        dictionary[id] = data
    # sort dictionary by date
    sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]["date"]))
    return sorted_dictionary

def get_prior_patent_list(patent_mapping, date):
    """returns list of patent_ids that have been published prior to *date*"""
    prior_patent_ls = []
    for key, value in patent_mapping.items():
        current_date = value["date"]
        if current_date < date:
            prior_patent_ls.append(key)
        else:
            break # since patent_mapping is sorted by date, we can exit (there are no more earlier patents)
    return prior_patent_ls

def calculate_tf(patent_mapping, patent_id, term):
    """receives patent_mapping, patent_id and the term of interest. Returns the frequency of *term* in the patent
    identified by *patent_id*"""
    filepath = patent_mapping[patent_id]["filepath"]
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.split()
    term_frequency = 0
    for token in text:
        if token == term:
            term_frequency += 1
    return term_frequency / len(text) # TODO: Should it be normalized?

def calculate_bidf(patent_mapping, patent_id, term, date):
    """Calculates the BIDF score for a given patent and term.
    High value: Most patents before *patent_id* do not contain word *term*
    Low value: Most patents before *patent_id* do contain word *term* 
    """
    prior_patents_ls = get_prior_patent_list(patent_mapping, date)
    documents_with_term_count = 0
    def process_patent(patent_id):
        filepath = patent_mapping[patent_id]["filepath"]
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if term in words:
                    return 1
        return 0
    with ThreadPoolExecutor() as executor:
        documents_with_term_count = sum(executor.map(process_patent, prior_patents_ls))
    return math.log(len(prior_patents_ls) / (1 + documents_with_term_count))

def calculate_patent_similarity(patent_mapping, patent_id_i, patent_id_j):
    """calculates the cosine similarity between patent i and patend j. 
    Steps:
    1) calculate TFBIDF for patent i and j, with t=min(t_i, t_j)
    2) create vectors W_i, W_j 
    3) calculate cosine similarity between W_i and W_j
    """
    min_date = min(patent_mapping[patent_id_i]["date"], patent_mapping[patent_id_j]["date"])

    with open(patent_mapping[patent_id_i]["filepath"], "r", encoding="utf-8") as f:
        text_i = f.read()
    text_i = text_i.split()
    terms_i = set(text_i)

    with open(patent_mapping[patent_id_j]["filepath"], "r", encoding="utf-8") as f:
        text_j = f.read()
    text_j = text_j.split()
    terms_j = set(text_j)    

    union = terms_i.union(terms_j)
    union = list(union) # change to list so it's iterable

    # calculate tfbidf for every term in union of words and store it in vector w
    w_i = np.zeros(len(union))
    w_j = np.zeros(len(union))
    for i in range(len(union)):
        term = union[i]
        tf_i = calculate_tf(patent_mapping, patent_id_i, term)
        bidf_i = calculate_bidf(patent_mapping, patent_id_i, term, min_date)
        w_i[i] = tf_i * bidf_i

        tf_j = calculate_tf(patent_mapping, patent_id_j, term)
        bidf_j = calculate_bidf(patent_mapping, patent_id_j, term, min_date)
        w_j[i] = tf_j * bidf_j

    # normalize vectors by dividing by L2 norm (euclidean norm)
    normalized_w_i = w_i / np.linalg.norm(w_i)
    normalized_w_j = w_j / np.linalg.norm(w_j)

    # calculate cosine similarity
    similarity = np.dot(normalized_w_i, normalized_w_j) / (np.linalg.norm(normalized_w_i) * np.linalg.norm(normalized_w_j))
    return similarity # TODO: Output should be between 0 and 1?!

def calculate_patent_similarity_parallel(patent_mapping, patent_id_i, patent_id_j):
    min_date = min(patent_mapping[patent_id_i]["date"], patent_mapping[patent_id_j]["date"])

    with open(patent_mapping[patent_id_i]["filepath"], "r", encoding="utf-8") as f:
        text_i = f.read()
    terms_i = set(text_i.split())

    with open(patent_mapping[patent_id_j]["filepath"], "r", encoding="utf-8") as f:
        text_j = f.read()
    terms_j = set(text_j.split())

    union = list(terms_i.union(terms_j))

    # Helper function to calculate w_i or w_j for a specific term
    def calculate_w(patent_id, term):
        tf = calculate_tf(patent_mapping, patent_id, term)
        bidf = calculate_bidf(patent_mapping, patent_id, term, min_date)
        return tf * bidf

    # Calculate w_i and w_j in parallel
    with ThreadPoolExecutor() as executor:
        w_i = np.array(list(executor.map(lambda term: calculate_w(patent_id_i, term), union)))
        w_j = np.array(list(executor.map(lambda term: calculate_w(patent_id_j, term), union)))

    # Normalize vectors
    normalized_w_i = w_i / np.linalg.norm(w_i)
    normalized_w_j = w_j / np.linalg.norm(w_j)

    # Calculate cosine similarity
    similarity = np.dot(normalized_w_i, normalized_w_j) / (np.linalg.norm(normalized_w_i) * np.linalg.norm(normalized_w_j))
    return similarity


def get_backward_document_list(patent_mapping, patent_id, num_years):
    """returns a list of patents *num_years* prior to document"""
    patent_date = patent_mapping[patent_id]["date"]
    prior_patent_ls = []
    for key, value in patent_mapping.items():
        if key == patent_id:
            continue
        current_patent_date = value["date"]
        if 0 < (patent_date - current_patent_date).days <= (num_years*365): # only include patents before *num_years*
            prior_patent_ls.append(key)
    return prior_patent_ls

def get_forward_document_list(patent_mapping, patent_id, num_years):
    """returns a list of documents (patents) *num_years* filed after the current document"""
    patent_date = patent_mapping[patent_id]["date"]
    prior_patent_ls = []
    for key, value in patent_mapping.items():
        if key == patent_id:
            continue
        current_patent_date = value["date"]
        if 0 < (current_patent_date - patent_date).days <= (num_years*365): # only include patents after *num_years*

            prior_patent_ls.append(key)
    return prior_patent_ls

def calculate_backward_similarity(patent_mapping, patent_id, backward_years):
    """calculates BS for a patent by adding the pairwise similarity of the patent and a set of prior patents filed in '
    the *backward_year*s prior to the current patent's filing"""
    backward_patents = get_backward_document_list(patent_mapping, patent_id, backward_years)
    backward_similarity = 0
    for backward_patent in backward_patents:
        backward_similarity += calculate_patent_similarity(patent_mapping, patent_id, backward_patent)
    return backward_similarity

def calculate_forward_similarity(patent_mapping, patent_id, forward_years):
    """calculates FS for a patent by adding the pairwise similarity of the patent and a set of successor patents 
    filed in the *forward_years* after the current patent's filing"""
    forward_patents = get_forward_document_list(patent_mapping, patent_id, forward_years)
    forward_similarity = 0
    for forward_patent in forward_patents:
        forward_similarity += calculate_patent_similarity(patent_mapping, patent_id, forward_patent)
    return forward_similarity

def calculate_patent_importance(patent_mapping, patent_id, backward_years=5, forward_years=10):
    """Returns higher values for patents that are novel and influential"""
    novelty = calculate_backward_similarity(patent_mapping, patent_id, backward_years)
    impact = calculate_forward_similarity(patent_mapping, patent_id, forward_years)
    return impact / novelty


if __name__=="__main__":
    textfolder_path = "test_preprocessed/"
    jsonfolder_path = "test_json/"
    patent_mapping = generate_json(textfolder_path, jsonfolder_path)

    # TODO: reduce time frame from 18.. - 19...

    patent_importance = calculate_patent_importance(patent_mapping, "14513")
    print(patent_importance)