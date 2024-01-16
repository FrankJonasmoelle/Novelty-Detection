import pandas as pd
import matplotlib.pyplot as plt


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