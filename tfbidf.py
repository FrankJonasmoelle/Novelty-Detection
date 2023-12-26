import math
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


SIMILARITY_DICT = defaultdict(dict) # dictionary that stores similarity scores between patents

def generate_json(textfolder_path, jsonfolder_path, start_year=1885, end_year=1928):
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

def get_prior_patent_list(patent_mapping, date):
    """returns list of patent ids that have been published prior to *date*"""
    prior_patent_ls = []
    for key, value in patent_mapping.items():
        current_date = value["date"]
        if current_date < date:
            prior_patent_ls.append(key)
        else:
            break # since patent_mapping is sorted by date, we can exit (there are no more earlier patents)
    return prior_patent_ls 

def get_patents_in_timerange(patent_mapping, start_year, end_year):
    """returns list of patent ids within a given time range"""
    ls = []
    for key, value in patent_mapping.items():
        current_date = value["date"] 
        if start_year <= current_date.year <= end_year:
            ls.append(key)
    return ls

def calculate_tf(patent_mapping, patent_id, term):
    """receives patent_mapping, patent_id and the term of interest. Returns the frequency of *term* in the patent
    identified by *patent_id*"""
    text = patent_mapping[patent_id]["text"]
    if len(text) == 0:
        return 0
    term_frequency = 0
    for token in text:
        if token == term:
            term_frequency += 1
    tf = term_frequency / len(text)
    return tf

def calculate_term_frequencies_per_patent(patent_mapping, start_year, end_year):
    """generate the following that is used to speed up the bidf calculation:
    {
        "1114": {"term_1": 192, "term_2": 185},
        "1119" {"term_1": 193, "term_2": 185},
          ...
    }
    It functions as a lookup dictionary for fast retrival of the number of patents that include a certain term 
    a specific date. Works 
    """
    global_term_count = {}
    term_count_per_patent = {}

    patents = get_patents_in_timerange(patent_mapping, start_year=start_year, end_year=end_year)
    num_prior_patents = 0
    for patent in patents:
        text = patent_mapping[patent]["text"]
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

def search_w_count(patent_mapping, patent, term, term_count_per_patent):
    """Helper function for retrival of the number of patents that include a certain term until a specific date. 
    It uses the *term_count_per_patent* dictionary to search for the last occurance of term *term* and 
    returns the accurate count used for the bidf calculation."""
    prevous_patent_ids_ls = get_prior_patent_list(patent_mapping, patent_mapping[patent]["date"]) #everything before id
    # search reversed list
    prevous_patent_ids_ls.reverse()
    for patent_id in prevous_patent_ids_ls:
        # check if term exists in term_count_per_patent for patent_id
        if term in term_count_per_patent[patent_id]["counts"]:
            w_count = term_count_per_patent[patent_id]["counts"][term]
            return w_count
    # if nothing is returned by now, the term does not exist before patent_id, i.e., count is 0
    return 0

def calculate_bidf_memoization(patent_mapping, term_count_per_patent, term, earlier_patent_id):
    """Calculates the BIDF score for a given patent and term using memoization. 
    High value: Most patents before *patent_id* do not contain word *term*
    Low value: Most patents before *patent_id* do contain word *term* 
    """
    num_patents_prior = term_count_per_patent[earlier_patent_id]["num_prior_patents"]
    try: 
        num_patents_prior_with_w = term_count_per_patent[earlier_patent_id]["counts"][term]
    except Exception as e: # exception will happen if earlier_patent does not include "term", in this case search in term_count_per_patent for previous count
        # search for it in previous patents in term_count_per_patent
        num_patents_prior_with_w = search_w_count(patent_mapping, earlier_patent_id, term, term_count_per_patent)
    try:
        bidf = math.log(num_patents_prior / (1 + num_patents_prior_with_w))
    except Exception as e: # exception will happen for the first document as there will be no prior documents
        print(e)
        return 0
    return bidf

def calculate_patent_similarity_memoization(patent_mapping, term_count_per_patent, patent_id_i, patent_id_j):
    """calculates the cosine similarity between patent i and patend j using memoization. 
    Steps:
    1) calculate TFBIDF for patent i and j, with t=min(t_i, t_j)
    2) create vectors W_i, W_j 
    3) calculate cosine similarity between W_i and W_j
    """
    min_date = min(patent_mapping[patent_id_i]["date"], patent_mapping[patent_id_j]["date"])
    if min_date == patent_mapping[patent_id_i]["date"]:
        earlier_patent_id = patent_id_i
    else:
        earlier_patent_id = patent_id_j

    text_i = patent_mapping[patent_id_i]["text"]
    terms_i = set(text_i)

    text_j = patent_mapping[patent_id_j]["text"]
    terms_j = set(text_j)    

    union = terms_i.union(terms_j)
    union = list(union) # change to list so it's iterable

    w_i = np.zeros(len(union))
    w_j = np.zeros(len(union))
    for i in range(len(union)):
        term = union[i]
        tf_i = calculate_tf(patent_mapping, patent_id_i, term)
        tf_j = calculate_tf(patent_mapping, patent_id_j, term)
        bidf_w = calculate_bidf_memoization(patent_mapping, term_count_per_patent, term, earlier_patent_id) # need to calculate it only once

        w_i[i] = tf_i * bidf_w # if patent i/j does not include term, value will be 0
        w_j[i] = tf_j * bidf_w

    normalized_w_i = w_i / np.linalg.norm(w_i)
    normalized_w_j = w_j / np.linalg.norm(w_j)

    # calculate cosine similarity
    similarity = np.dot(normalized_w_i, normalized_w_j) / (np.linalg.norm(normalized_w_i) * np.linalg.norm(normalized_w_j))
    return similarity

def calculate_bidf_naive(patent_mapping, patent_id, term, date):
    """Calculates the BIDF score for a given patent and term.
    High value: Most patents before *patent_id* do not contain word *term*
    Low value: Most patents before *patent_id* do contain word *term* 
    """
    prior_patents_ls = get_prior_patent_list(patent_mapping, date)
    documents_with_term_count = 0
    for patent in prior_patents_ls:
        text = patent_mapping[patent]["text"]
        if term in text:
            documents_with_term_count += 1
    try:
        bidf = math.log(len(prior_patents_ls) / (1 + documents_with_term_count))
    except Exception as e:
        print(e)
    return bidf

def calculate_patent_similarity_naive(patent_mapping, patent_id_i, patent_id_j):
    """calculates the cosine similarity between patent i and patend j. 
    Steps:
    1) calculate TFBIDF for patent i and j, with t=min(t_i, t_j)
    2) create vectors W_i, W_j 
    3) calculate cosine similarity between W_i and W_j
    """
    min_date = min(patent_mapping[patent_id_i]["date"], patent_mapping[patent_id_j]["date"])

    text_i = patent_mapping[patent_id_i]["text"]
    terms_i = set(text_i)

    text_j = patent_mapping[patent_id_j]["text"]
    terms_j = set(text_j)    

    union = terms_i.union(terms_j)
    union = list(union) # change to list so it's iterable

    # calculate tfbidf for every term in union of words and store it in vector w
    w_i = np.zeros(len(union))
    w_j = np.zeros(len(union))
    for i in range(len(union)):
        term = union[i]
        tf_i = calculate_tf(patent_mapping, patent_id_i, term)
        tf_j = calculate_tf(patent_mapping, patent_id_j, term)

        bidf = calculate_bidf_naive(patent_mapping, patent_id_i, term, min_date) # use bfidf from earlier patent

        w_i[i] = tf_i * bidf
        w_j[i] = tf_j * bidf

    # normalize vectors by dividing by L2 norm (euclidean norm)
    normalized_w_i = w_i / np.linalg.norm(w_i)
    normalized_w_j = w_j / np.linalg.norm(w_j)

    # calculate cosine similarity
    similarity = np.dot(normalized_w_i, normalized_w_j) / (np.linalg.norm(normalized_w_i) * np.linalg.norm(normalized_w_j))
    if math.isnan(similarity):
        print("similarity is nan")
        similarity = 0
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

def calculate_backward_similarity(patent_mapping, term_count_per_patent, patent_id, backward_years):
    """calculates BS for a patent by adding the pairwise similarity of the patent and a set of prior patents filed in '
    the *backward_year*s prior to the current patent's filing"""
    backward_patents = get_backward_document_list(patent_mapping, patent_id, backward_years)
    backward_similarity = 0
    for backward_patent_id in backward_patents:
        if backward_patent_id not in SIMILARITY_DICT[patent_id]:  
            similarity = calculate_patent_similarity_memoization(patent_mapping, term_count_per_patent, patent_id, backward_patent_id)
            SIMILARITY_DICT[patent_id][backward_patent_id] = similarity
            SIMILARITY_DICT[backward_patent_id][patent_id] = similarity
        else:
            similarity = SIMILARITY_DICT[patent_id][backward_patent_id] # retrieve similarity from dictionary
        backward_similarity += similarity
        print(f"BS between {patent_id} and {backward_patent_id} is {similarity}")
    # scale backward similarity
    try:
        backward_similarity = backward_similarity / len(backward_patents)
    except Exception as e:
        print(e)
        backward_similarity = 0
    print(f"backward similarity for patent {patent_id}: {backward_similarity}")
    return backward_similarity

def calculate_forward_similarity(patent_mapping, term_count_per_patent, patent_id, forward_years):
    """calculates FS for a patent by adding the pairwise similarity of the patent and a set of successor patents 
    filed in the *forward_years* after the current patent's filing"""
    forward_patents = get_forward_document_list(patent_mapping, patent_id, forward_years)
    forward_similarity = 0
    for forward_patent_id in forward_patents:
        if forward_patent_id not in SIMILARITY_DICT[patent_id]: 
            similarity = calculate_patent_similarity_memoization(patent_mapping, term_count_per_patent, patent_id, forward_patent_id)
            SIMILARITY_DICT[patent_id][forward_patent_id] = similarity # store score to be able to retrieve it
            SIMILARITY_DICT[forward_patent_id][patent_id] = similarity
        else:
            similarity = SIMILARITY_DICT[patent_id][forward_patent_id] # retrieve similarity from dictionary
        forward_similarity += similarity
        print(f"FS between {patent_id} and {forward_patent_id} is {similarity}")
    # scale forward similarity
    try:
        forward_similarity = forward_similarity / len(forward_patents)
    except Exception as e:
        print(e)
        forward_similarity = 0
    print(f"forward similarity for patent {patent_id}: {forward_similarity}")
    return forward_similarity

def calculate_patent_importance(patent_mapping, term_count_per_patent, patent_id, backward_years=5, forward_years=10):
    """Returns higher values for patents that are novel and influential"""
    novelty = calculate_backward_similarity(patent_mapping, term_count_per_patent, patent_id, backward_years)
    impact = calculate_forward_similarity(patent_mapping, term_count_per_patent, patent_id, forward_years)
    importance = impact/novelty
    return novelty, impact, importance

def get_swedish_population_by_year(start_year, end_year):
    """returns a dic that maps the year to the population size of Sweden"""
    df_pop_sweden = pd.read_excel("population_sweden.xlsx", skiprows=4)
    df_pop_sweden = df_pop_sweden[:250]
    condition1 = df_pop_sweden.iloc[:, 0] >= start_year 
    condition2 = df_pop_sweden.iloc[:, 0] <= end_year
    combined_condition = condition1 & condition2
    filtered_df = df_pop_sweden[combined_condition]
    year = filtered_df.iloc[:, 0]
    population = filtered_df.iloc[:, 1]
    year_population_mapping = dict(zip(year, population))
    return year_population_mapping

def plot_breakthrough_patents(input_file="results.csv"):
    """Plots the number of breakthrough patents per capita. Breakthrough patents are those that fall in the
    top 10 percent of the unconditional distribution of our importance measure."""
    results = pd.read_csv(input_file)
    # sort by importance value
    sorted_df = results.sort_values(by='importance', ascending=False)
    # get top 10% importances
    top_10_percentile = int(len(sorted_df) * 0.1)
    df_top_10_percent = sorted_df.head(top_10_percentile)
    df_top_10_percent['year'] = pd.to_datetime(df_top_10_percent['year'])
    df_top_10_percent['year'] = df_top_10_percent['year'].dt.year
    # group by year and count number of patents in this year
    top_percentile_patents_per_year = df_top_10_percent.groupby('year').size()
    # scale by Swedish population
    top_percentile_patents_per_year = top_percentile_patents_per_year.to_dict()
    population_by_year = get_swedish_population_by_year(1890, 1918)
    scaled_breakthrough_patents_by_year = {}
    for year, num_breakthrough_patents in top_percentile_patents_per_year.items():
        population = population_by_year[year]
        num_breakthrough_patents = top_percentile_patents_per_year[year]
        scaled_patent_count = (num_breakthrough_patents / population) * 1000
        scaled_breakthrough_patents_by_year[year] = scaled_patent_count
    # plot
    years = scaled_breakthrough_patents_by_year.keys()
    count_per_year = scaled_breakthrough_patents_by_year.values()
    plt.figure(figsize=(15, 6))
    plt.plot(years, count_per_year)
    plt.xlabel('Year')
    plt.ylabel('Number of Patents')
    plt.ylim(0, max(count_per_year)*1.2)
    plt.title('Number of Breakthrough Patents per Year')
    plt.xticks(rotation=45, ha='right')  # Adjust the rotation angle as needed
    plt.savefig("breakthrough_patents_per_year.png")
    plt.show()

def plot_num_patents(patent_mapping, start_year=1890, end_year=1918):
    """plots the total number of patents per 1.000 people (scaled by swedish population)"""
    patents_per_year = {}
    for key, value in patent_mapping.items():
        year = value["date"].year
        if str(year) not in patents_per_year:
            patents_per_year[str(year)] = 1
        else:
            patents_per_year[str(year)] += 1
    population_by_year = get_swedish_population_by_year(start_year, end_year)
    scaled_patents_by_year = {}
    for year, population in population_by_year.items():
        num_patents = patents_per_year[str(year)]
        scaled_patent_count = (num_patents / population) * 1000
        scaled_patents_by_year[year] = scaled_patent_count
    
    # plotting
    years = list(scaled_patents_by_year.keys())
    count_per_year = list(scaled_patents_by_year.values())
    plt.figure(figsize=(10, 6))
    plt.plot(years, count_per_year)
    plt.xlabel('Year')
    plt.ylabel('Number of patents per 1,000 people')
    plt.title('Total patent count per capita')
    plt.xticks(rotation=45, ha='right')  # Adjust the rotation angle as needed
    plt.savefig("patents_per_capita.png")
    plt.show()

def load_existing_results(path):
    if os.path.exists(path):
        df_results = pd.read_csv(path, index_col=0)
        results = df_results.to_dict(orient='index')
        results = {str(key): value for key, value in results.items()} # convert keys to strings
    else:
        results = {}
    return results

def main(textfolder_path, jsonfolder_path, output_path="results.csv"):
    # generate mapping
    print("generating mapping")
    patent_mapping = generate_json(textfolder_path, jsonfolder_path)

    # generate dictionary that keeps track of term frequencies and number of previous patents for a given patent
    print("generating term count dictionary")
    term_count_per_patent = calculate_term_frequencies_per_patent(patent_mapping, start_year=1885, end_year=1928)

    # start_year: 1890
    # end_year: 1918
    patents_in_range = get_patents_in_timerange(patent_mapping, start_year=1890, end_year=1918)

    print("loading previous results")
    results = load_existing_results(output_path)

    print("calculating patent_importance")
    count = 0
    for patent in patents_in_range:
        if patent not in results: 
            novelty, impact, importance = calculate_patent_importance(patent_mapping, term_count_per_patent, patent, backward_years=5, forward_years=10)
            print(f"Scores calculated for patent {patent}: Novelty: {novelty}, Impact: {impact}, Importance: {importance}")
            scores = {"novelty": novelty, "impact": impact, "importance": importance, "year": patent_mapping[patent]["date"]}
            results[patent] = scores
        else:
            print(f"Score of patent {patent} already calculated")
        if count % 10 == 0: # after every 10 patents, add results to csv
            df = pd.DataFrame.from_dict(results, orient='index')
            df.to_csv(output_path)
        count += 1
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(output_path)
    return
    
if __name__=="__main__":
    textfolder_path = "../data_preprocessed/"
    jsonfolder_path = "../json/"

    main(textfolder_path, jsonfolder_path)