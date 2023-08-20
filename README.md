# README

Logion is a system for detecting errors in ancient and medieval Greek works. 
Here we document the training procedure for a premodern Greek BERT model, as well as how to utilize it for error detection. We provide BERT training code for those interested in replicating our training or using their own data; for those interested in using our premodern Greek BERT model out of the box, which was trained on over 70 million words of premodern Greek, we make this model available along with instructions to use it here: https://huggingface.co/cabrooks/LOGION-base. 

This model can also be fine-tuned on specific works of interest to better suit a given task or trained from scratch (see train_example.py).

For details on using the code in this repo to generate error detection reports on works of your choice, follow the instructions in [this document](https://docs.google.com/document/d/1CEVQ_oLJX4Cwy9zUQM9CVorauXeQHe8R41pMjQrc4vs/edit?usp=sharing).

For more information about our group, see https://www.logionproject.princeton.edu. 

Barbara Graziosi<sup>1</sup>, Johannes Haubold<sup>1</sup>, Charlie Cowen-Breen<sup>2</sup>, Creston Brooks<sup>3</sup>
<i><br>
<sup>1</sup> Department of Classics, Princeton University [barbara.graziosi@princeton.edu](mailto:barbara.graziosi@princeton.edu); [jhaubold@princeton.edu](mailto:jhaubold@princeton.edu) <br>
<sup>2</sup> Department of Pure Mathematics and Mathematical Statistics, University of Cambridge [wcc4@princeton.edu](mailto:wcc4@princeton.edu) / [wc320@cam.ac.uk](mailto:wc320@cam.ac.uk)<br>
<sup>3</sup> Department of Computer Science, Princeton University [cabrooks@princeton.edu](mailto:cabrooks@princeton.edu)
</i>
<br>

## System requirements

It is recommended, but not required, that your system has a GPU in order to perform inference with Logion. On a system with Python >=3.8.8;
Conda >=4.10.1, one can execute<br/>
```
>> conda create --name logion pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
>> conda activate logion
```
to initialize the environment. 

### System recommendations for training
Logion was trained on a research computing cluster with 2.8 GHz Intel Ice Lake nodes for several days. If you intend to fine-tune, it's recommended that your processor has at least 128 GB of memory and a GPU. With a Nvidia K80 / T4 (standard on Google Colab), beam search should take no more than 10 seconds for spans of up to 10 tokens, with current specifications.
