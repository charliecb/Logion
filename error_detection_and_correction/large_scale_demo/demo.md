# Large-scale Implementation of Error Detection and Correction

Here we demonstrate a possible implementation of Logion on a large-scale GPU-enabled computing cluster. For a high-level overview of the intended behavior of each script, see `README.md` in the above directory. In what follows, we use 40 nodes of a SLURM cluster, with each node powered by a 2.8 GHz Intel Ice Lake CPU with a single 1.41 GHz A100 GPU.

## Step 1: Configure environment
On a system with Python >=3.8.8; Conda >=4.10.1, execute
```
>> conda create --name logion pytorch torchvision torchaudio polyleven cudatoolkit=11.3 -c pytorch
>> conda activate logion
```
In what follows, we will assume use of a SLURM cluster.

## Step 2: Check Levenshtein transducer
If you are using the WordPiece tokenizer located on HuggingFace at `cabrooks/LOGION-50k_wordpiece` with 50k tokens, no changes need to be made and you may proceed to Step 3.

However, if you choose to use a custom tokenizer, you will need to execute `lev.py` on your desired tokenizer with all Levenshtein distances of interest. `lev.py` will precompute the Levenshtein distance between all pairs of tokens in the vocabulary, and record a stored `vocab_size` x `vocab_size` array of booleans indicating the pairs of tokens which are within the specified Levenshtein distance threshold. In `build_report.py`, this array will be loaded from the filesystem and used to filter acceptable suggestions, enormously saving time to build reports.

## Step 3: Data and model specification
### Option 1: Model trained via cross-validation on 5-way data split
Store the directories for each of your models in a single parent directory (say, `your_model_directory`) under the names `model1`, `model2`, etc. In `build_report.py`, set
```
model_path = f'your_model_directory/model{split_num}'
```
Store your entire dataset in a single `.txt` file, in an order consistent with the numbering of the models, and point `data_path` to this file.

### Option 2: Single dataset and model
In `build_report.py`, set `model_path` to your BERT model filepath, `tokenizer_path` to your tokenizer filepath, and `data_path` to your tokenized dataset filepath (i.e. line-separated collections of 512 or fewer tokens). Additionally, change line 125 to
```
data_split = data
```

## Step 4: SLURM parallelization
To accelerate the computation of CCR, one may design a SLURM workload such that each node processes only a small number of lines (again, each roughly 512 tokens) of the dataset. In the example given here, each node is designed to process 100 lines (although this may be changed by passing the `num_pars` flag to `build_report.py`. With the hardware specified and each node processing 100 lines, each node has a runtime of approximately 32 hours.

Parallelization is enabled by several command line arguments `build_report.py` accepts:
* `--num_pars` is the number of lines processed by a single node; recommendation: 100 ("pars" = lines)
* `--start_at` is the line number within the data split of interest at which to start; recommendation: 100, 200, 300, ...
* `--split_num` is the number of the dataset split which is used; recommendation: 1, 2, 3, 4, 5, as described in Step 3: Option 1.
* `--lev` is the Levenshtein distance threshold between suggestions and transmissions; recommendation: 1 or 2
Output files are stored in `./{lev}/{split_num}/{start_at}`.

In the demonstration provided here, the corpus is divided into a 5-way data split, each fifth of which is further divided into 8 sections each of 100 lines or fewer, for a total of 40 mini-datasets. 40 SLURM batch script files are used queue 40 such instances of `build_report.py`, each searching for and providing suggested corrections for errors occuring in its niche of the corpus. Each instance of `build_report.py` will process at most 100 lines, so we expect jobs to take at most 32 hours with the hardware specified.

The files named `report{lev}.{split_num}.{start_at}.slurm` function as batch scripts to send each such job to the SLURM queue. To execute the code as intended, run
```
sbatch report{lev}.{split_num}.{start_at}.slurm
```
for every `lev` = 1; `split_num` = 1, 2, 3, 4, 5; `start_at` = 0, 100, 200, 300, 400, 500, 600, 700.

Each instance of `build_report.py` will generate its own output sub-directory, each containing files which record the transmitted text, suggested text, and CCR of every word in the corpus.

## Step 5: Graphical report generation
This final step is the most straightforward. Simply execute
```
python graphical_report_generation.py
```
to produce a visualization of the final report in HTML. Ensure that `header.txt` and `footer.txt` are in the same directory as `graphical_report_generation.py`. If one wishes to modify the number of flags produced, change `num_flags` in `graphical_report_generation.py`. If one wishes to generate a report consisting of only flags from a certain sub-directory&mdash;such as the one corresponding to data split 3, for example&mdash;one can change `base_directory` in `graphical_report_generation.py`&mdash;in this case, to `./3`; note that this also changes the output directory which contains `finalreport.htm` accordingly.

Finally, standard web browsers can be used to convert the output HTML to PDF. To do this, open `finalreport.htm` in e.g. Safari, and select File -> Print -> PDF. For a good rule of thumb, with the 50k tokenizer provided on HuggingFace at `cabrooks/LOGION-50k_wordpiece`, 100 flags means roughly 60 PDF pages, and 500 flags means roughly 300 PDF pages.

We emphasize that these reports are intended only to provide inspiration for domain experts, and must be interpreted by philologists before formal conclusions are drawn.
