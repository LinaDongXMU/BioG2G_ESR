# BioG2G_ESR
Unified Graph-to-Graph Retrosynthesis and Enzyme Sequence Recommendation for Biocatalytic Design

## 2. Enzyme Sequence Recommender
This tool, enzyme_sequence_recommender.py, is designed to bridge the gap between retrosynthetic prediction and enzymatic validation. Once you have obtained a predicted reaction (the transformation of reactants to products), you can input it into this script to identify the most suitable enzyme sequences from a preprocessed database.

### 2.1 Workflow
The recommendation process follows a two-stage validation strategy to ensure high-accuracy results:

Phase 1: Fingerprint Filtering Uses reaction fingerprints and Tanimoto similarity to quickly retrieve the top candidates that share a similar chemical transformation logic.

Phase 2: SMARTS Substructure Ranking Performs a deep-dive verification by matching the local reaction center (SMARTS template) of the database entries against your query molecules. This ensures the enzyme's specific catalytic site is compatible with your substrate.

### 2.2 Usage
1. Prepare your Query
Organize your retrosynthetic output into a standard Reaction SMILES format: ReactantA.ReactantB>>ProductA.

2. Preprocess the Database
Ensure your JSON database (e.g., SHJT_EnzymeMap_cleaned_smarts2_recursion.json) is in the same directory. On the first run, the script will generate a .pkl cache to speed up future queries.

3. Run the Script
You can call the recommend_enzymes function within the script:


```
from enzyme_sequence_recommender import recommend_enzymes

# Example: Amide hydrolysis or your retrosynthesis result
query = "CC1=CC=C(S(N(CC=C)CC=C)(=O)=O)C=C1>>CC1=CC=C(S(N2CC=CC2)(=O)=O)C=C1"
recommend_enzymes(query, db, top_k=10)
```

### 2.3 Output Categories
High Confidence: Candidates where the structural template matches the query reaction center perfectly.

Medium Confidence: Candidates with high fingerprint similarity but where the specific template match is weaker or not applicable.
