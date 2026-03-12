import os
import ast
from rdkit import Chem
from tqdm import tqdm 

def convert_mapped_smiles_to_canonical(
    mapped_smiles: str,
    keep_atom_map: bool = False,
    remove_hydrogens: bool = True,
    kekulize: bool = True,
    isomeric: bool = True
) -> str:
    """
    Converts SMILES with atom mapping numbers to canonical SMILES
    """
    try:
        mol = Chem.MolFromSmiles(mapped_smiles)
        if mol is None:
            return None
        
        if kekulize:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        
        if not keep_atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        
        if remove_hydrogens:
            mol = Chem.RemoveHs(mol)
        
        canonical_smiles = Chem.MolToSmiles(
            mol, 
            canonical=True,
            isomericSmiles=isomeric,
            kekuleSmiles=kekulize
        )
        
        return canonical_smiles
    
    except Exception:
        return None


def predict_using_BioG2G(smiles, model_path):
    input_file = "../dataset/plus/infer_input.txt"
    output_file = "process.txt"
    
    try:
        with open(input_file, "w+") as f:
            f.write(smiles)
        
        os.system(f'sh infer_BioG2G.sh {model_path} > {output_file}')
        
        with open(output_file, "r") as f:
            results = f.read()
        
        input1 = results.split('\n')[-2]
        input1 = ast.literal_eval(input1)
        
        result_dict = {
            'reactants': [], 
            'scores': [], 
            'types': [], 
            'templates': []
        }
        
        for i in input1:
            for j, k in enumerate(i):
                cleaned_reactant = convert_mapped_smiles_to_canonical(
                    k, 
                    keep_atom_map=False,
                    remove_hydrogens=True,
                    kekulize=True,
                    isomeric=True
                )
                
                if cleaned_reactant is not None:
                    result_dict['reactants'].append(cleaned_reactant)
                    result_dict['scores'].append((1 - int(j) * 0.1)/5.5)
                    result_dict['types'].append('B/C')
                    result_dict['templates'].append(None)
                else:
                    print(f"Skipping SMILES that cannot be converted: {k}")
        
        # 🔹 Added: Deduplication logic
        unique_reactants = []
        unique_scores = []
        unique_types = []
        unique_templates = []
        
        seen = set()
        for smi, score, t, tmp in zip(
            result_dict['reactants'],
            result_dict['scores'],
            result_dict['types'],
            result_dict['templates']
        ):
            if smi not in seen:
                seen.add(smi)
                unique_reactants.append(smi)
                unique_scores.append(score)
                unique_types.append(t)
                unique_templates.append(tmp)
        
        result_dict['reactants'] = unique_reactants
        result_dict['scores'] = unique_scores
        result_dict['types'] = unique_types
        result_dict['templates'] = unique_templates
        # 🔹 End of deduplication

        return result_dict
    
    finally:
        files_to_remove = [
            input_file,
            output_file,
            "../dataset/plus/infer_result.txt",
            "../dataset/plus/args.txt"
        ]
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e.strerror}")

def process_predictions(model_path, src_file="test_src.txt", out_file="test_pred.txt"):
    """
    Batch call the BioG2G model to predict each line of test_src.txt,
    outputting test_pred.txt (10 lines of predicted reactants for each input).
    """
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Input file does not exist: {src_file}")

    with open(src_file, "r", encoding="utf-8") as f_src:
        smiles_list = [line.strip() for line in f_src if line.strip()]

    print(f"Read {len(smiles_list)} molecules for prediction")
    print(f"Starting batch prediction, model path: {model_path}")

    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx, smi_eg in enumerate(tqdm(smiles_list, desc="Predicting", unit="mol")):
            try:
                results = predict_using_BioG2G(smi_eg, model_path)
                reactants = results.get("reactants", [])
                
                # Deduplicate while maintaining order
                seen = set()
                unique_reactants = []
                for r in reactants:
                    if r not in seen:
                        seen.add(r)
                        unique_reactants.append(r)

                # Take top 10
                top10 = unique_reactants[:10]

                # If fewer than 10, pad with empty lines
                while len(top10) < 10:
                    top10.append("")

                # Write to file, each prediction result occupies 10 lines
                for r in top10:
                    f_out.write(r + "\n")

            except Exception as e:
                # Write 10 empty lines if inference fails
                print(f"Inference failed for item {idx+1}: {repr(e)}")
                for _ in range(10):
                    f_out.write("\n")

    print(f"\n All predictions completed!")
    print(f" Output file: {out_file}")
    print(f" Expected lines: {len(smiles_list)*10}")

# Usage Example
# if __name__ == "__main__":
#    model_path = "/path/to/checkpoint.pt"
#    smi_eg = "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
#    results = predict_using_BioG2G(smi_eg, model_path)
 #   print(results)

if __name__ == "__main__":
    model_path = "../checkpoint/main/plus/checkpoint_best.pt"
    smi = "CC(C)C(=O)C1=CC=CC=C1"
    results = predict_using_BioG2G(smi, model_path)
    print(results)
    # process_predictions(model_path, src_file="test_src.txt", out_file="test_pred.txt")