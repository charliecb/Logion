# Error Detection and Correction
Here we provide code, tutorials, and demonstrations for the error detection and correction feature of Logion. We provide both small-scale and large-scale implementations, depending on the hardware available to the user. For users intending to detect errors of the highest quality within large corpora, we recommend the large-scale implementation, suitable for parallelization on computer clusters with multiple GPUs available. For users intending to see a minimalistic demonstration of the algorithm, or without access to a GPU, we recommend the small-scale implementation.

## Logion pipeline
Here we descibe the high-level procedure implemented in both the small-scale and large-scale implementations. For specifics of each implementation, see the respective sub-directories.

As described in [1], the error detection and correction pipeline consists of three stages. In the first stage, a transformer is trained with the standard masked language model (MLM) objective on the corpus of interest. Here we assume that this has already been done; guidance for performing this procedure can be found in the parent directory. In the second and third stages, on which we focus here, the chance-confidence ratio (CCR) [1] of every word is computed, and corresponding MLM determined suggestions are generated.

### Report building

This procedure is called **report building**, and is performed in `build_report.py`, located in each of the sub-directories. `build_report.py` imports functionality from `logion.py`; the two files should be kept in the same directory. 

`build_report.py` generates five output files:
1. `transmissions.txt` contains each word of the text as it appears in the dataset, i.e. the transmitted text
2. `suggestions.txt` contains the top MLM suggestion for each word, when restricted to the appropriate Levenshtein distance, separated by asterisks (so as to allow for suggestions which split one word into multiple)
3. `wordratios.txt` contains the CCRs of each word, separated by commas
4. `wordchances.txt` contains the chances of each word, separated by commas (see [1])
5. `wordconfidences.txt` contains the confidences of each word, separated by commas (see [1])

Each of the files above contains one line for each section of text which may be simultaneously processed by the MLM (typically, 512 tokens or less).

### Graphical report generation

Once the CCR of every word has been calculated by `build_report.py` and the five output files described above have been generated, the report still must be **graphically generated**. This involves ranking all words in ascending order of their CCR, and presenting the output to the user visually.

This procedure is performed in `graphical_report_generation.py`, located in each of the sub-directories. `graphical_report_generation.py` searches `base_directory` (defined in file) for all sub-directories which contain each of the five files described above. It then recombines all such files into a single collection of all transmissions, suggestions, wordratios, wordchances, and wordconfidences, respectively. This separation and subsequent recombination is useful for parallelization, as explored in the large-scale implementation.

`graphical_report_generation.py` then treats every `(transmission, suggestion, wordratio, wordchance, wordconfidence)` tuple as a flag, the CCR of which is given by `wordratio`. Flags are ranked in ascending order of their CCR and stored.

To visualize the output, for each flag, the corresponding line of text is determined, and each word is color-coded by its CCR for ease of visulization. Red words are "suspicious", and green words are "safer". The top `num_flags` (defined in file) flags are included in the report.

Amalgamated flags are visualized in HTML. **Styling is provided in `header.txt` and `footer.txt` and must remain in the same directory as `graphical_report_generation.py`.**

Finally, `graphical_report_generation.py` will output `finalreport.htm`, which contains a stylized representation of the report. Standard web browsers can be used to convert `finalreport.htm` to a PDF for ease of use.
