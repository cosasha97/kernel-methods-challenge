# kernel-methods-challenge

## Context
Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

## Generate our results
To generate our results, you only need to run the following command:
```
python start.py
```
**Warning:** the computation of the different kernels will take a very long time.

## Content
* "data/": contains the embedded data and the raw strings
* "results/": contains csv files recording the results of our different trainings
* "scripts/": contains the scripts of the different methods and kernels
  *  "spectrum.py": spectrum kernel
  *  "mismatch.py": mismatch kernel
  *  "substring.py": substring kernel
  *  "models.py": script containing the code for:
    * a class 'KernelMethod' (parent class any each kernel method)
    * Kernel Ridge Regression
    * Kernel SVM (version with and without bias)
    
