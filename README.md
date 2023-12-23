# Novelty Detection in Swedish Patent Data

The aim of this project is to use natural language processing to identify novel and valuable
technologies within Swedish historical patent data.

Patent impact calculation is implemented according to the method described by Kelly et al. (2021). According to the authors, important patents can be described the following:

- **Novel** patents are distinct from their predecessors

- **Impactful** patents influence future scientific advances, manifested as high similarity with subsequent innovations/patents

- An **important** patent is both novel **and** impactful

- (An **important** patent is **dissimilar** to previous patents and **similar** to future patents)


### Steps

1) Calculate **novelty**: calculate backward similarity (BS) between a patent $p$ and all patents published 5 years prior to $p$.

$$
BS_{j}^{\tau} = \frac{1}{\left| \mathcal{B}_{j, \tau} \right|} \sum_{i \in \mathcal{B}_{j, \tau}} p_{j, i}
$$

for patents $i$ and $j$.


2) Calculate **impact**: calculate forward similarity (FS) between a patent $p$ and all patents published 10 years after $p$

$$
FS_{j}^{\tau} = \frac{1}{\left| \mathcal{F}_{j, \tau} \right|} \sum_{i \in \mathcal{F}_{j, \tau}} p_{j, i}
$$

for patents $i$ and $j$.

3) Calculate **importance**: calculate ratio between FS and BS -> higher value if novelty and impact are high // BS is low and FS is high

$$
q_{j}^{\tau} = \frac{FS_{j}^{\tau}}{BS_{j}^{\tau}}
$$

### The similarity score is based on TFIBDF, a variatin of TFIDF: 
For a patent $p$ and a term $w$, the TFBIDF score is calculated the following:

$$
    TFBIDF_{w, p} = TF_{w, p} \cdot BIDF_{w}
$$

where 

$$
    TF_{w, p} = \frac{c_{p,w}}{\sum_{k}{}c_{p,k}}
$$

and

$$
    BIDF_{w, p} = \log \left( \frac{ \text{ num patents prior to } p}{1 + \text{ num documents prior to } p \text{ that include term } w} \right)
$$

To calculate the similarity $p_{i, j}$ between patent $i$ and $j$, the TFBIDF score is calculated for each term $w$ in patents $i$ and $j$.
The result is stored in the vectors $w_i$ and $w_j$.
Finally, the cosine similarity between the two vectors is calculated.