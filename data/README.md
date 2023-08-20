# README

The data found in raw_data_resources.zip was made available to us by Pranaydeep Singh, which he assembled as part of his work on [A Pilot Study for BERT Language Modelling and Morphological Analysis](https://aclanthology.org/2021.latechclfl-1.15.pdf). 

We also include cleaned_data_example, which is the result of our cleaning, organizing, and splitting this data. The resulting train/test files are newline separated with a training example on each line. Each of these training examples contains <= 512 tokens (using [this tokenizer](https://huggingface.co/cabrooks/LOGION-base)).

Logion is a system for detecting errors in ancient and medieval Greek works. 
Here we document the training procedure for a premodern Greek BERT model, as well as how to utilize it for error detection. We provide BERT training code for those interested in replicating our training or using their own data; for those interested in using our premodern Greek BERT model out of the box, which was trained on over 70 million words of premodern Greek, we make this model available along with instructions to use it here: https://huggingface.co/cabrooks/LOGION-base. 

This model can also be fine-tuned on specific works of interest to better suit a given task or trained from scratch (see train_example.py).

For details on using the code in this repo to generate error detection reports on works of your choice, follow the instructions in [this document](https://docs.google.com/document/d/1CEVQ_oLJX4Cwy9zUQM9CVorauXeQHe8R41pMjQrc4vs/edit?usp=sharing).

For more information about our group, see https://www.logionproject.princeton.edu. 
