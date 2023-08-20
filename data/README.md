# README

The data found in raw_data_resources.zip was made available to us by Pranaydeep Singh, which he assembled as part of his work on [A Pilot Study for BERT Language Modelling and Morphological Analysis](https://aclanthology.org/2021.latechclfl-1.15.pdf). 

We also include cleaned_data_example, which is the result of our cleaning, organizing, and splitting this data. The resulting train/test files are newline separated with a training example on each line. We maximized the amount of punctuation-separated text in each example, up to a limit of 512 tokens using [this tokenizer](https://huggingface.co/cabrooks/LOGION-base). 

Data for premodern Greek faces a specific problem which needs to be addressed. The best online database is the Thesaurus Linguae Graecae ([TLG](https://stephanus.tlg.uci.edu/)). It is not open access (unlike the best databases for other ancient languages, e.g. Latin). We are grateful to the TLG Director for providing us with some of the data we used for our models; we were instructed, however, that it cannot be disseminated further, because of the license currently restricting access to the TLG. The global archive of premodern texts is an important reservoir of linguistic and cultural diversity which should be accurately digitized and made freely available. For now, we make available the models we trained along with all training data that can be disseminated;

For details on using the code in this repo to generate error detection reports on works of your choice, follow the instructions in [this document](https://docs.google.com/document/d/1CEVQ_oLJX4Cwy9zUQM9CVorauXeQHe8R41pMjQrc4vs/edit?usp=sharing).

For more information about our group, see https://www.logionproject.princeton.edu. 
