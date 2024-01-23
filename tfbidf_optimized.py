import math
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter

def generate_json(textfolder_path, jsonfolder_path, start_year=1885, end_year=1929):
    """generates mapping between the patent id and additional data
    {
        "1":{
            "filepath": "data/1.txt",
            "date": datetime.datetime(1900, 3, 17, 0, 0)
            "text": ["det", "är, "ett", "exempel"]
        },
        "2":{
            "filepath": "data/2.txt",
            "date": datetime.datetime(1901, 3, 17, 0, 0)
            "text": ["det", "är, "ett", "exempel"]
        },
        "3":{
            "filepath": "data/3.txt",
            "date": datetime.datetime(1902, 3, 17, 0, 0)
            "text": ["det", "är, "ett", "exempel"]
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
        # get meta information
        dpk = json_file["dpk"]
        ipc = json_file["ipc"]
        date = datetime.strptime(date, '%Y-%m-%d') # convert to datetime object
        if start_year <= date.year <= end_year: # only consider patent if it's grant date is within range of interest
            with open(text_filepath, "r", encoding="utf-8") as f:
                text = f.read()
            text = text.split()
            if len(text) == 0: # skip if text is empty
                continue
            data = {"filepath": text_filepath,
                    "date": date,
                    "text": text,
                    "dpk": dpk,
                    "ipc": ipc}
            dictionary[id] = data
        else:
            continue
    # sort dictionary by date
    sorted_dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]["date"]))
    return sorted_dictionary

def generate_tf_mapping(start_date, end_date):
    """Generates dictionary that stores the term frequency scores for each term in a patent."""
    tf_mapping = {}
    patents = get_patents_in_timerange(start_date, end_date)
    for patent in patents:
        text = PATENT_MAPPING[patent]["text"]
        term_counts = Counter(text)  
        term_frequencies = {term: count/len(text) for term, count in term_counts.items()}
        tf_mapping[patent] = term_frequencies
    return tf_mapping

def calculate_num_term_occurances(total_start_year, start_year, end_year):
    """generates a dict that keeps track of the number of occurances of unique words until each patent document.
    Used for more efficient bidf calculations:
    {
        "1114": {'counts': {'undanskaffas': 1,
                            'vagna': 24,
                            'cylind': 82,
                            'förut': 237,
                            'äfv': 522,
                            ...
                            },
                'num_prior_patents': 871},
        "1115": {'counts': {....},
                 'num_prior_patents': ...}
    }
    """
    global_term_count = {}
    term_count_per_patent = {}
    num_prior_patents = 0

    # Process patents from total_start_year to start_year
    if total_start_year != start_year: # for 1885, skip it
        patents = get_patents_in_timerange(start_year=total_start_year, end_year=start_year-1)
        for patent in patents:
            text = PATENT_MAPPING[patent]["text"]
            local_vocab = set(text)
            for term in local_vocab:
                if term in global_term_count:
                    global_term_count[term] += 1
                else:
                    global_term_count[term] = 1
            num_prior_patents += 1

    # Process patents from start_year to end_year
    patents = get_patents_in_timerange(start_year=start_year, end_year=end_year)
    for patent in patents:
        text = PATENT_MAPPING[patent]["text"]
        local_vocab = set(text)
        for term in local_vocab:
            if term in global_term_count:
                global_term_count[term] += 1
            else:
                global_term_count[term] = 1
        term_count_per_patent[patent] = {"counts": global_term_count.copy(), "num_prior_patents": num_prior_patents}
        num_prior_patents += 1
    return term_count_per_patent

def get_total_vocab():
    """returns total vocabulary of patents"""
    total_vocab = set()
    for _, data in PATENT_MAPPING.items():
        text = data["text"]
        vocab = set(text)
        total_vocab.update(vocab)
    return list(total_vocab)

def levenshtein_distance(term_1, term_2):
    """returns levenshtein distance between two terms"""
    if len(term_1) > len(term_2):
        term_1, term_2 = term_2, term_1
    distances = range(len(term_1) + 1)
    for index2, char2 in enumerate(term_2): 
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(term_1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

def find_similar_words(min_edit_distance=1, max_edit_distance=2):
    """goes through whole vocabulary and finds similar words for each word. Sorts the words alphabetically and only
    checks the next n words to speed up computations."""
    n = 30 # check only the next n words after a word (speeds up computation)
    total_vocab = get_total_vocab()
    sorted_total_vocab = sorted(total_vocab) # sort alphabetically to check only similar words
    similar_words_dict = {word: [] for word in sorted_total_vocab} # store the words and the similar ones for inspection
    for index, term_1 in enumerate(sorted_total_vocab):
        next_n_terms = sorted_total_vocab[index+1:index+n+1] # only check next 20 terms
        for term_2 in next_n_terms:
            edit_distance = levenshtein_distance(term_1, term_2)
            # only consider words with small edit distance (smaller edit distance for smaller words)
            if (len(term_1) < 5 and edit_distance < min_edit_distance) or \
                (edit_distance <= max_edit_distance and len(term_1) > 5 and len(term_2) > 5):
                # add them to the same word 
                if not term_1 in similar_words_dict:
                    similar_words_dict[term_1] = [term_2]
                else:
                    similar_words_dict[term_1].append(term_2)      
    return similar_words_dict

def map_similar_words(path="word_mapping.csv"):
    """generates a mapping between every word in the vocabulary to the closest word based on the levensthein distance.
    Example output:
    {
        'absorberingsappara': 'absorberingsappara',
        'absorberingsapparat': 'absorberingsappara',
        'absorberinusappara': 'absorberingsappara',
        'absorberingsförmåga': 'absorberingsförmåga',
        'absorberingskam': 'absorberingskam',
        'absorberingskammare': 'absorberingskammare',
        'absorberingskamrarne': 'absorberingskammare',
    }
    """
    if os.path.exists(path):
        df_mapping = pd.read_csv(path, index_col=0)
        results = df_mapping.to_dict(orient='index')
        results = {str(key): value for key, value in results.items()} # convert keys to strings
        return mapping
    
    similar_words_dict = find_similar_words()
    mapping = {}
    for term, similar_word_ls in similar_words_dict.items():
        if term not in mapping:
            mapping[term] = term
        reference_word = mapping[term]
        for similar_word in similar_word_ls:
            if similar_word not in mapping:
                mapping[similar_word] = reference_word
    # save it to not have to recompute it everytime
    df = pd.DataFrame.from_dict(mapping, orient='index')
    df.to_csv(path) 
    return mapping

def apply_word_mapping(word_mapping):
   """applies vocabulary mapping generated by *map_similar_words* to the actual data in *PATENT_MAPPING*"""
   for patent_id, _ in PATENT_MAPPING.items(): 
      text = PATENT_MAPPING[patent_id]["text"]
      replaced_words = [word_mapping.get(word, word) for word in text] # returns value from mapping if word is found, else the original word
      PATENT_MAPPING[patent_id]["text"] = replaced_words
         
# functions to cut words below certain frequency
def calculate_total_term_frequencies(start_year=1885, end_year=1939):
    """calculates frequency for each term in patents within time range"""
    vocab_tf = {}
    patents = get_patents_in_timerange(start_year, end_year)
    for patent in patents:
        text = PATENT_MAPPING[patent]["text"]
        for term in text:
            if term in vocab_tf:
                vocab_tf[term] += 1
            else:
                vocab_tf[term] = 1
    return vocab_tf

def get_low_frequency_words(min_frequency=2):
    """returns set of words that occurr less than *min_frequency*"""
    vocab_tf = calculate_total_term_frequencies()
    words_to_remove = set()
    for key, value in vocab_tf.items():
        if value < min_frequency:
            words_to_remove.add(key)
    return words_to_remove

def remove_words(patent_id, text, words_to_remove):
    """helper function for *remove_low_frequency_words_parallel*"""
    new_text = [word for word in text if word not in words_to_remove]
    return patent_id, new_text

def remove_low_frequency_words_parallel(min_frequency=2):
    """Iterates through whole *PATENT_MAPPING* and removes words that occurr less than *min_frequency*"""
    words_to_remove = get_low_frequency_words(min_frequency)
    
    with multiprocessing.Pool() as pool:
        results = pool.starmap(remove_words, [(patent_id, data["text"], words_to_remove) 
                                              for patent_id, data in PATENT_MAPPING.items()])
    for patent_id, new_text in results:
        PATENT_MAPPING[patent_id]["text"] = new_text

def get_prior_patent_list(patent):
    """Returns list of patent_ids that have been published prior to *patent*"""
    if patent in PATENT_TO_INDEX:
        index = PATENT_TO_INDEX[patent]
        return PATENT_LIST[:index]
    return []

def get_sorted_patent_list():
    """Returns list of patent ids, sorted by date."""
    patent_list = []
    for key, _ in PATENT_MAPPING.items():
        patent_list.append(key)
    return np.array(patent_list)

def get_patents_in_timerange(start_year, end_year):
    """returns list of patent ids within a given time range"""
    ls = []
    for key, value in PATENT_MAPPING.items():
        current_date = value["date"] 
        if start_year <= current_date.year <= end_year:
            ls.append(key)
    return ls

def calculate_tf(patent_id, term):
    """receives patent_mapping, patent_id and the term of interest. Returns the frequency of *term* in the patent
    identified by *patent_id*"""
    if term not in TF_MAPPING[patent_id]:
        return 0
    else:
        return TF_MAPPING[patent_id][term]

def calculate_bidf(term, earlier_patent_id):
    """Calculates the BIDF score for a given patent and term using memoization. 
    High value: Most patents before *patent_id* do not contain word *term*
    Low value: Most patents before *patent_id* do contain word *term* 
    """
    num_patents_prior = len(get_prior_patent_list(earlier_patent_id))
    if term in TERM_COUNT_PER_PATENT[earlier_patent_id]["counts"]:
        num_patents_prior_with_w = TERM_COUNT_PER_PATENT[earlier_patent_id]["counts"][term]
    else:
        num_patents_prior_with_w = 0
    try:
        bidf = math.log(num_patents_prior / (1 + num_patents_prior_with_w))
    except Exception as e: # exception will happen for the first document as there will be no prior documents
        print(e)
        return 0
    return bidf

def get_top_n_tfbidf_scores(patent_id, n=5):
    """returns n terms with highest tfbidf scores for a given patent"""
    text = PATENT_MAPPING[patent_id]["text"]
    vocab = np.array(list(set(text)))
    tfbidf_arr = np.zeros(len(vocab))   
    for i in range(len(vocab)):
        term = vocab[i]
        tf = calculate_tf(patent_id, term)
        bidf = calculate_bidf(term, patent_id)
        tfbidf_w = tf*bidf
        tfbidf_arr[i] = tfbidf_w
    # get max n tfbidf terms
    indices = np.argpartition(tfbidf_arr, -n)[-n:] # gets indices of n largest values
    indices = indices[np.argsort(tfbidf_arr[indices])[::-1]] # sorts the indices in descending order
    top_n_tfbidf_scores = vocab[indices] # map to vocabulary
    return top_n_tfbidf_scores

def calculate_patent_similarity(patent_id_i, patent_id_j):
    """calculates the cosine similarity between patent i and patend j using memoization. 
    Steps:
    1) calculate TFBIDF for patent i and j, with t=min(t_i, t_j)
    2) create vectors W_i, W_j 
    3) calculate cosine similarity between W_i and W_j
    """
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
        bidf_w = calculate_bidf(term, earlier_patent_id) # need to calculate it only once

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
    for backward_patent_id in backward_patents:
        similarity = calculate_patent_similarity(patent_id, backward_patent_id)
        backward_similarity += similarity
    # scale backward similarity
    try:
        backward_similarity = backward_similarity / len(backward_patents)
    except Exception as e:
        print(e)
        backward_similarity = 0
    return backward_similarity

def calculate_forward_similarity(patent_id, forward_years):
    """calculates FS for a patent by adding the pairwise similarity of the patent and a set of successor patents 
    filed in the *forward_years* after the current patent's filing"""
    forward_patents = get_forward_document_list(patent_id, forward_years)
    forward_similarity = 0
    for forward_patent_id in forward_patents:
        similarity = calculate_patent_similarity(patent_id, forward_patent_id)
        forward_similarity += similarity
    # scale forward similarity
    try:
        forward_similarity = forward_similarity / len(forward_patents)
    except Exception as e:
        print(e)
        forward_similarity = 0 
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
    """Worker function for importance calculation for multiprocessing"""
    novelty, impact, importance = calculate_patent_importance(patent, backward_years=5, forward_years=10)
    top_tfbidf_terms_ls = get_top_n_tfbidf_scores(patent, n=10)
    print(f"Scores calculated for patent {patent}: Novelty: {novelty}, Impact: {impact}, Importance: {importance}")
    scores = {"novelty": novelty, "impact": impact, "importance": importance, "year": PATENT_MAPPING[patent]["date"],
              "dpk": PATENT_MAPPING[patent]["dpk"],
              "ipc": PATENT_MAPPING[patent]["ipc"],
              "top_tfbidf_terms": top_tfbidf_terms_ls}
    return patent, scores

def main(output_path="results.csv"):
    patents_in_range = get_patents_in_timerange(start_year=1890, end_year=1929)
    results = load_existing_results(output_path)

    patents_to_process = [patent for patent in patents_in_range if patent not in results]

    # Set up multiprocessing
    pool_size = multiprocessing.cpu_count() 
    count = 0
    with multiprocessing.Pool(processes=pool_size) as pool:
        for patent, scores in pool.imap_unordered(worker, patents_to_process):
            results[patent] = scores
            if count % 5 == 0: 
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
PATENT_MAPPING = generate_json(textfolder_path, jsonfolder_path, start_year=1885, end_year=1939)

# preprocessing
print("preprocessing step: mapping similar words together")
word_mapping = map_similar_words()
print("preprocessing step: applying new word mapping")
apply_word_mapping(word_mapping) # updates text in PATENT_MAPPING
# cut words under threshold
print("preprocessing step: removing low frequency words")
remove_low_frequency_words_parallel(min_frequency=2) # updates text in PATENT_MAPPING

print("generating term count dictionary")
TERM_COUNT_PER_PATENT = calculate_num_term_occurances(1885, 1885, 1910) # TODO: Until 1929

print("generating tf mapping")
TF_MAPPING = generate_tf_mapping(start_date=1885, end_date=1939)

print("generating sorted patent list")
PATENT_LIST = get_sorted_patent_list()
PATENT_TO_INDEX = {patent: idx for idx, patent in enumerate(PATENT_LIST)}


if __name__=="__main__":
    main()
