import math
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from functools import lru_cache
import multiprocessing


def generate_json(textfolder_path, jsonfolder_path, start_year=1885, end_year=1928):
    """generates mapping between the patent id and additional data
    {
        "1":{
            "filepath": "data/1.txt",
            "date": datetime.datetime(1900, 3, 17, 0, 0)
            "text": ["det", "är, "en", "exempel"]
        },
        "2":{
            "filepath": "data/2.txt",
            "date": datetime.datetime(1901, 3, 17, 0, 0)
            "text": ["det", "är, "en", "exempel"]
        },
        "3":{
            "filepath": "data/3.txt",
            "date": datetime.datetime(1902, 3, 17, 0, 0)
            "text": ["det", "är, "en", "exempel"]
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
        date = json_file["grant_date"]
        if date is None:
            continue # skip patent
        date = datetime.strptime(date, '%Y-%m-%d') # convert to datetime object
        if start_year <= date.year <= end_year: # only consider patent if it's grant date is within range of interest
            with open(text_filepath, "r", encoding="utf-8") as f:
                text = f.read()
            text = text.split()
            if len(text) == 0: # skip if text is empty
                continue
            data = {"filepath": text_filepath,
                    "date": date,
                    "text": text}
            dictionary[id] = data
        else:
            continue
    # sort dictionary by date
    sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]["date"]))
    return sorted_dictionary

def get_prior_patent_list(date):
    """returns list of patent_ids that have been published prior to *date*"""
    prior_patent_ls = []
    for key, value in PATENT_MAPPING.items():
        current_date = value["date"]
        if current_date < date:
            prior_patent_ls.append(key)
        else:
            break # since patent_mapping is sorted by date, we can exit (there are no more earlier patents)
    return prior_patent_ls 

def get_patents_in_timerange(start_year, end_year):
    ls = []
    for key, value in PATENT_MAPPING.items():
        current_date = value["date"] 
        if start_year <= current_date.year <= end_year:
            ls.append(key)
    return ls

@lru_cache(maxsize=500000)  
def calculate_tf(patent_id, term):
    """receives patent_mapping, patent_id and the term of interest. Returns the frequency of *term* in the patent
    identified by *patent_id*"""
    text = PATENT_MAPPING[patent_id]["text"]
    if len(text) == 0:
        return 0
    term_frequency = 0
    for token in text:
        if token == term:
            term_frequency += 1
    try:
        tf = term_frequency / len(text)
    except Exception as e:
        print(e)
        tf = 0
    return tf

def calculate_term_frequencies_per_patent(start_year, end_year):
     # generate the following dict:
    # {
    #     "1114": {"term_1": 192, "term_2": 185},
    #     "1119" {"term_1": 193, "term_2": 185},
    #       ...
    # }
    global_term_count = {}
    term_count_per_patent = {}

    patents = get_patents_in_timerange(start_year=start_year, end_year=end_year)
    num_prior_patents = 0
    for patent in patents:
        text = PATENT_MAPPING[patent]["text"]
        local_vocab = set(text)
        # Num of patents prior to the patent that include term w
        local_term_count = {}
        for term in local_vocab:
            # Increase counter in global_term_count if vocab already exists, else add it:
            if term not in global_term_count:
                global_term_count[term] = 1
            else:
                global_term_count[term] += 1
            # Retrieve count of term from global_term_count and add it to local_term_count
            local_term_count[term] = global_term_count[term]
        term_count_per_patent[patent] = {"counts": local_term_count, "num_prior_patents": num_prior_patents}
        num_prior_patents += 1
    return term_count_per_patent

def search_w_count(patent, term):
    prevous_patent_ids_ls = get_prior_patent_list(PATENT_MAPPING[patent]["date"]) #everything before id
    # search reversed list
    prevous_patent_ids_ls.reverse()
    for patent_id in prevous_patent_ids_ls:
        # check if term exists in term_count_per_patent for patent_id
        if term in TERM_COUNT_PER_PATENT[patent_id]["counts"]:
            w_count = TERM_COUNT_PER_PATENT[patent_id]["counts"][term]
            return w_count
    # if nothing is returned by now, the term does not exist before patent_id, i.e., count is 0
    return 0

@lru_cache(maxsize=5000000)  
def calculate_bidf_memoization(term, earlier_patent_id):
    num_patents_prior = TERM_COUNT_PER_PATENT[earlier_patent_id]["num_prior_patents"]
    try: 
        num_patents_prior_with_w = TERM_COUNT_PER_PATENT[earlier_patent_id]["counts"][term]
    except Exception as e: # exception will happen if earlier_patent does not include "term", in this case search in term_count_per_patent for previous count
        # search for it in previous patents in term_count_per_patent
        num_patents_prior_with_w = search_w_count(earlier_patent_id, term)
    try:
        bidf = math.log(num_patents_prior / (1 + num_patents_prior_with_w))
    except Exception as e: # exception will happen for the first document as there will be no prior documents
        print(e)
        return 0
    return bidf

@lru_cache(maxsize=5000000) 
def calculate_patent_similarity_memoization(patent_id_i, patent_id_j):
    min_date = min(PATENT_MAPPING[patent_id_i]["date"], PATENT_MAPPING[patent_id_j]["date"])
    if min_date == PATENT_MAPPING[patent_id_i]["date"]:
        earlier_patent_id = patent_id_i
    else:
        earlier_patent_id = patent_id_j

    text_i = PATENT_MAPPING[patent_id_i]["text"]
    terms_i = set(text_i)

    text_j = PATENT_MAPPING[patent_id_j]["text"]
    terms_j = set(text_j)    

    union = terms_i.union(terms_j)
    union = list(union) # change to list so it's iterable

    w_i = np.zeros(len(union))
    w_j = np.zeros(len(union))

    for i in range(len(union)):
        term = union[i]
        tf_i = calculate_tf(patent_id_i, term)
        tf_j = calculate_tf(patent_id_j, term)
        bidf_w = calculate_bidf_memoization(term, earlier_patent_id) # need to calculate it only once

        w_i[i] = tf_i * bidf_w # if patent i/j does not include term, value will be 0
        w_j[i] = tf_j * bidf_w

    normalized_w_i = w_i / np.linalg.norm(w_i)
    normalized_w_j = w_j / np.linalg.norm(w_j)

    # calculate cosine similarity
    similarity = np.dot(normalized_w_i, normalized_w_j) / (np.linalg.norm(normalized_w_i) * np.linalg.norm(normalized_w_j))
    if math.isnan(similarity):
        similarity = 0
    return similarity

def get_backward_document_list(patent_id, num_years):
    """returns a list of patents *num_years* prior to document"""
    patent_date = PATENT_MAPPING[patent_id]["date"]
    prior_patent_ls = []
    for key, value in PATENT_MAPPING.items():
        if key == patent_id:
            continue
        current_patent_date = value["date"]
        if 0 < (patent_date - current_patent_date).days <= (num_years*365): # only include patents before *num_years*
            prior_patent_ls.append(key)
    return prior_patent_ls

def get_forward_document_list(patent_id, num_years):
    """returns a list of documents (patents) *num_years* filed after the current document"""
    patent_date = PATENT_MAPPING[patent_id]["date"]
    prior_patent_ls = []
    for key, value in PATENT_MAPPING.items():
        if key == patent_id:
            continue
        current_patent_date = value["date"]
        if 0 < (current_patent_date - patent_date).days <= (num_years*365): # only include patents after *num_years*
            prior_patent_ls.append(key)
    return prior_patent_ls

def calculate_backward_similarity(patent_id, backward_years):
    """calculates BS for a patent by adding the pairwise similarity of the patent and a set of prior patents filed in '
    the *backward_year*s prior to the current patent's filing"""
    backward_patents = get_backward_document_list(patent_id, backward_years)
    backward_similarity = 0
    count = 0
    for backward_patent_id in backward_patents:
        similarity = calculate_patent_similarity_memoization(patent_id, backward_patent_id)
        backward_similarity += similarity
        if count % 100 == 0:
            print(f"BS between {patent_id} and {backward_patent_id} is {similarity}")
        count += 1
    # scale backward similarity
    try:
        backward_similarity = backward_similarity / len(backward_patents)
    except Exception as e:
        print(e)
        backward_similarity = 0
    print(f"backward similarity for patent {patent_id}: {backward_similarity}")
    return backward_similarity

def calculate_forward_similarity(patent_id, forward_years):
    """calculates FS for a patent by adding the pairwise similarity of the patent and a set of successor patents 
    filed in the *forward_years* after the current patent's filing"""
    forward_patents = get_forward_document_list(patent_id, forward_years)
    forward_similarity = 0
    count = 0
    for forward_patent_id in forward_patents:
        similarity = calculate_patent_similarity_memoization(patent_id, forward_patent_id)
        forward_similarity += similarity
        if count % 100 == 0:
            print(f"FS between {patent_id} and {forward_patent_id} is {similarity}")
        count += 1
    # scale forward similarity
    try:
        forward_similarity = forward_similarity / len(forward_patents)
    except Exception as e:
        print(e)
        forward_similarity = 0
    print(f"forward similarity for patent {patent_id}: {forward_similarity}")
    return forward_similarity

def calculate_patent_importance(patent_id, backward_years=5, forward_years=10):
    """Returns higher values for patents that are novel and influential"""
    novelty = calculate_backward_similarity(patent_id, backward_years)
    impact = calculate_forward_similarity(patent_id, forward_years)
    importance = impact/novelty
    return novelty, impact, importance

def load_existing_results(path):
    if os.path.exists(path):
        df_results = pd.read_csv(path, index_col=0)
        results = df_results.to_dict(orient='index')
        results = {str(key): value for key, value in results.items()} # convert keys to strings
    else:
        results = {}
    return results

def worker(patent):
    # worker function to enable multiprocessing
    novelty, impact, importance = calculate_patent_importance(patent, backward_years=5, forward_years=10)
    print(f"Scores calculated for patent {patent}: Novelty: {novelty}, Impact: {impact}, Importance: {importance}")
    scores = {"novelty": novelty, "impact": impact, "importance": importance, "year": PATENT_MAPPING[patent]["date"]}
    return patent, scores

def main(output_path="results.csv"):
    patents_in_range = get_patents_in_timerange(start_year=1890, end_year=1918)
    results = load_existing_results(output_path)

    # Prepare a list of patents that need processing
    patents_to_process = [patent for patent in patents_in_range if patent not in results]

    # Set up multiprocessing pool
    pool_size = multiprocessing.cpu_count() 
    count = 0
    with multiprocessing.Pool(processes=pool_size) as pool:
        for patent, scores in pool.imap_unordered(worker, patents_to_process):
            results[patent] = scores
            if count % 10 == 0:  # Adjust this number based on your needs
                print("trying to store...")
                df = pd.DataFrame.from_dict(results, orient='index')
                df.to_csv(output_path)
            count += 1

    # Save final results
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(output_path)

# use global variables to allow for multiprocessing
print("generating patent mapping")
textfolder_path = "../data_preprocessed/"
jsonfolder_path = "../json/"
PATENT_MAPPING = generate_json(textfolder_path, jsonfolder_path)

print("generating term count dictionary")
TERM_COUNT_PER_PATENT = calculate_term_frequencies_per_patent(start_year=1885, end_year=1928)


if __name__=="__main__":
    main()