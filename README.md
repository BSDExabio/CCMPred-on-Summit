# CCMpred-on-Summit
CCMpred is a tool for learning Protein Residue-Residue Contacts from Correlated Mutations predicted based on a Markov Random Field pseudo-likelihood maximization. (https://github.com/soedinglab/CCMpred)

**Build on Summit**

```bash
# Clone the git repository

# Checkout a specific branch 
git checkout <branch_name>
# Load needed modules
module load gcc cuda cmake/3.15.2
# Confiuger and build
cmake 
make

```

**Run on Summit**

```bash
# Example
Using a single sequence file
jsrun -n 1 -a 1 -c 1 -b packed:1 -g 1 ./bin/ccmpred  ./example/2ID0A.aln out.mat

Using a set of sequence files
export OMP_NUM_THREADS=4
jsrun --smpiargs="-disable_gpu_hooks" -n 1 -r 1 -a 1 -g 1 -c 4 -d packed -b rs ./bin/ccmpred  -f ./example/list.txt

Here, the list.txt has names of four sequence files - a line per file name as shown bellow,

./example/1atzA.aln
./example/2JEED.aln
./example/3CVEB.aln
./example/3IV1F.aln
```
