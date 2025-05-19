# Negative Ties Highlight Hidden Extremes in Social Media Polarization

Open source data and code for the research paper:

> E. Candellone,* S. A. Babul,* Ö. Togay, A. Bovet, and J. Garcia-Bernardo

Negative Ties Highlight Hidden Extremes in Social Media Polarization 

Pre-print: [https://arxiv.org/abs/2501.05590](https://arxiv.org/abs/2501.05590)

*shared first authors

## Contents of the repository

* `/bertopic/`: BERTopic intermediate results and model specifications
* `/data/`: CA and SHEEP emebddings and network files
* `/figures/`: paper figures
* `/hsbm/`: TM-hSBM intermediate results and model specifications
* `/ideology_twitter/`: validation with Twitter data and [PoliticalWatch](https://politicalwatch.es/en)
* `/notebooks/`
    * `1_topic_modelling.ipynb`: script to perform BERTopic and TM-hSBM topic modelling
    * `2_compare_hsbm_bert.ipynb`: comparison of the two methods to have robust topics
    * `3a_create_attitudes.ipynb`: create network embeddings using SHEEP and CA
    * `3b_sheep_null_model.ipynb`: null model to compare SHEEP and CA
    * `4_figures_paper.ipynb`: code to reproduce the figures of the paper
* `/src/`
    * `create_snapshot.py`: code to extract and clean data from scraped webpage.
    * `topicmodelling.py` helper functions for topic modelling.
    * `meneame.py`, `s3_create_attitudes.py`: helper functions for creating embeddings.




## Instructions

1. Create conda environment:

```
conda env create -f polarization.yml
conda activate polarization
```

2. Run the notebooks



## Contact

* Corresponding authors: [Elena Candellone](mailto:candellone.elena@gmail.com) and [Shazia Babul](mailto:shazia.babul@maths.ox.ac.uk)
