# Small-scale Implementation of Error Detection and Correction
Here we demonstrate a possible implementation of Logion on a personal computer without a GPU, noting that this hardware limitation may decrease output quality. For a high-level overview of the intended behavior of each script, see README.md in the above directory. In what follows, we use a computer with a 2.4 GHz Intel Broadwell CPU and 4 GB RAM.

## Step 1: Configure environment
On a system with Python >=3.8.8; Conda >=4.10.1, execute
```
>> conda create --name logion pytorch torchvision torchaudio polyleven cudatoolkit=11.3 -c pytorch
>> conda activate logion
```
In what follows, we will assume use of a SLURM cluster.

## Step 2: Download Levenshtein filter
From [Dropbox](https://www.dropbox.com/scl/fo/367ca4oabb2iwkydswfyg/h?rlkey=edsdb6b00bviltfo3a4bqefey&dl=0), download the directory `lev_maps` and store it in the same parent directory as `build_sample_report.py`, as indicated by the mock file system within `small_scale_demo`.

## Step 3: Data and model specification
For the purposes of our demonstration, we use as data a sample from the works of Michael Psellos, provided in `sample.txt`. After running this demo, the report generated will flag "η" within "η δε γαρ", which is a genuine instance of scribal error, previously undiscovered to the best of our knowledge [2]. On principle, however, one can edit `sample.txt` with any text of interest, provided it is within the token limit.

As a model, we use the pretrained BERT model stored on https://huggingface.co/cabrooks/LOGION-50k_wordpiece, as described in [1].

## Step 4: Report building
Execute
```
python build_sample_report.py -lev 1
```
to build the report. This should take less than 10 seconds with a GPU, and less than 5 minutes without.

## Step 5: Graphical report generation
Execute
```
python graphical_report_generation.py
```
to produce a visualization of the final report in HTML. Ensure that `header.txt` and `footer.txt` are in the same directory as `graphical_report_generation.py`. If one wishes to modify the number of flags produced, change `num_flags` in `graphical_report_generation.py`. If one wishes to generate a report consisting of only flags from a certain sub-directory&mdash;such as the one corresponding to data split 3, for example&mdash;one can change `base_directory` in `graphical_report_generation.py`&mdash;in this case, to `./3`; note that this also changes the output directory which contains `finalreport.htm` accordingly.

Finally, standard web browsers can be used to convert the output HTML to PDF. To do this, open `finalreport.htm` in e.g. Safari, and select File -> Print -> PDF. For a good rule of thumb, with the 50k tokenizer provided on HuggingFace at `cabrooks/LOGION-50k_wordpiece`, 100 flags means roughly 60 PDF pages, and 500 flags means roughly 300 PDF pages.

We emphasize that these reports are intended only to provide inspiration for domain experts, and must be interpreted by philologists before formal conclusions are drawn.

# References
[1] Charlie Cowen-Breen, Creston Brooks, Johannes Haubold, Barbara Graziosi. 2023. Logion: Machine-Learning Based Detection and Correction of Textual Errors in Greek Philology. *To appear in ACL 2023 Workshop (ALP).*

[2] B. Graziosi, J. Haubold, C. Cowen-Breen, and
C. Brooks. 2023. Machine learning and the future of
philology: A case study. TAPA, 153(1):253–284.
