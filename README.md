# CCMpred-on-Summit
CCMpred is a tool for learning Protein Residue-Residue Contacts from Correlated Mutations predicted based on a Markov Random Field pseudo-likelihood maximization. (https://github.com/soedinglab/CCMpred)

**Build on Summit**

```bash
# Laod needed modules
module load gcc cuda cmake/3.15.2
# Confiuger and build
cmake -DWITH_OMP=off
make

```
