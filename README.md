# Change Point Detection
LMU CIS 2020 master thesis

### 1. Scrappers
- settings (minor changes to prevent being blocked)
    - CONCURRENT_REQUESTS = 10
    - DOWNLOAD_DELAY =2 

#### Workflow
Get the list of rennovations sites.

- in `scrapper/`

```
scrapy crawl renovation_list -o scrapped_files/renovation_list_20200323.json
```

- clean the scrapped renovation list 

```
python clean_data/clean_scrapped.py --renovation_list --input scrapper/scrapped_files/renovation_list_20200323.json --output scrapper/scrapped_files/cleaned_renovation_list.json
```

- scrape the renovation hotels content

```
python clean_data/clean_scrapped.py --renovation_content --input scrapper/scrapped_files/renovation_hotels_content.jl --output scrapper/data/indexed_rennovation_hotel_content_20200323.json
```



#### Troubleshooting
* 429 too many request:
    - https://stackoverflow.com/a/48344232



### 2. clean query output 

- Transform json files from queried data to csv 

```
 python clean_data/inspect_query_data.py --transform --input query_outputs_all/ --output cleand_query_output_csv/
```

- Get id clusters.

```
python clean_data/clean_scrapped.py --renovation_file data/indexed_rennovation_hotel_content_20200323.json --api_result scrapper/scrapped_files/api_requests_hotels_indexed.jl --get_cluster_ids --output data/get_id_clusters_20200326.json
```
### 3. CPD models

#### ruptures:
- run in batch: (--binseg or --window)
```
python experimented_cpd/ruptures/rupture_model.py --input data/cleand_query_output_csv/ --output experimented_cpd/rupture_results/ --binseg --batch
```

### 4. Topic Modeling

* process_corpus (using all queried data)

`python topic_modeling/scripts/preprocessing.py --process_corpus`


### 5. Unsupervised Aspect Extraction with Attention
* preprocess:
    - generate training data from ty data (csv files)
        - 1 mio sentences
     - generate test files for inference.
* word2vec:
    - gensim generate word embedding model for training data
        - window=10, emb-size=200, worker=4
* training:
    - `python train.py --emb-name ../preprocessed_data/ty/w2v_embedding --domain ty --vocab-size 70000`
    - negative factor = 5
* infer:
    - ` python infer.py --domain ty --vocab-size 78215`


### 6. aspect based sentiment analysis
In repo `sentiment_analysis_experiment/MAMS-for-ABSA`:
- `python make_term_test_data_from_ty.py`
- `python test_ty.py`
- `python post_process.py`


### Experiences:
- nltk is more efficient than TextBlob, using nltk in preprocessing texts.
- noun phrases extraction 
    - noun chunks from Spacy include `a` and `the` articles.
    - noun phrases from TextBlob (CornllExtractor) -> best quality, very slow, cannot be parallelized 
    
 



## fields for matching hotels in database.

- goal: find the cluster_id
- fields:
    - name
    - description
    - score (100)
    - rating
    - modified_on
    - created_on
    - trust_score
    - address  (a dictionary)
        * city
        * fax 
        * coordinates 
            * "(-82.4547259,27.9423426)"
        * zip
        * district
        * webpage
        * country ("United States)
        * state ("Florida")
        * phone
            * +1 813-769-8300
        * street
        * email
        * country_code : "US"
    - starts: 3
    - relevance : 1
    - deleted_on : null

- to match the database, the fields in hopitality
    - latitude
    - longitude
    - hotelName
    - address (street, region zip, country)
    - phone
    - website
    
    
    
### setup
* jupyter lab kernel

`python -m ipykernel install --user --name cpd --display-name "cpd"`

* nltk
```python
import nltk
nltk.download('stopwords')
```

* spacy english corpus
```terminal
python3 -m spacy download en

spacy.load("en")
```



### tips

* explore streamlit:
    https://medium.com/analytics-vidhya/building-a-twitter-sentiment-analysis-app-using-streamlit-d16e9f5591f8
    
### reference
#### sentiment analysis
- text classification 
    https://github.com/ThilinaRajapakse/simpletransformers
