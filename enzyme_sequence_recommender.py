import json 
import sys
import os
import time
import pickle
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdChemReactions

# ============================================================================
# Basic Reaction Parsing and Fingerprint Functions
# ============================================================================
def parse_reaction_smiles(rxn_smiles):
    """Parses reactions in 'A.B>>C.D' format"""
    try:
        reactants_smiles, products_smiles = rxn_smiles.split('>>')
        reactants = tuple(Chem.MolFromSmiles(s) for s in reactants_smiles.split('.'))
        products = tuple(Chem.MolFromSmiles(s) for s in products_smiles.split('.'))
        if None in reactants or None in products:
            return None, None
        return reactants, products
    except Exception:
        return None, None


def calculate_reaction_fingerprint(rxn_smiles):
    """Calculates Difference Fingerprint for the reaction"""
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        fp = AllChem.CreateDifferenceFingerprintForReaction(rxn)
        return fp
    except Exception as e:
        print(f"[WARN] Fingerprint calculation failed: {rxn_smiles}, Error: {e}")
        return None


# ============================================================================
# [MOD] Template Validity: Validate template against its ref_rxn using substructure matching
# ============================================================================
def validate_template_against_reference(ref_rxn_smiles, template_smarts, db_index=None):
    """
    Checks if the template can at least 'find the reaction center' on its own reference reaction
    (Prevents bad templates from entering high-confidence matches)
    """
    ref_reactants, ref_products = parse_reaction_smiles(ref_rxn_smiles)
    if not ref_reactants or not ref_products:
        return False

    try:
        ok, score = check_template_match(
            ref_reactants,
            ref_products,
            template_smarts,
            allow_reverse=False,
            verbose=False,
            score_threshold=0.5  # Internal validation threshold is slightly more relaxed
        )
        if not ok and db_index is not None:
            print(f"[WARN] Template inconsistent with reference or too weak, db_index={db_index}, score={score:.3f}")
        return ok
    except Exception:
        return False


# ============================================================================
# Option B: Substructure matching based on SMARTS local reaction center (Robust)
# ============================================================================
def _compute_template_match_score_one_direction(query_mols, tpl_reactants):
    """
    Computes template match score in one direction (e.g., Template.reactants vs Query.reactants):
    - For each template reactant fragment (tpl_r):
        * Find a query_mol that matches it (substructure match)
        * Match Degree = matched atoms / template atoms
    - Total Score = Minimum of all template fragment match degrees (bottleneck determines score)
    """
    per_tpl_scores = []

    for tpl_r in tpl_reactants:
        smarts = Chem.MolToSmarts(tpl_r)
        tpl_mol = Chem.MolFromSmarts(smarts)
        if tpl_mol is None:
            return 0.0

        tpl_n_atoms = tpl_mol.GetNumAtoms() or 1
        best_ratio = 0.0

        for qmol in query_mols:
            matches = qmol.GetSubstructMatches(tpl_mol)
            if not matches:
                continue
            # If there's a match, it means tpl_r can be embedded in this qmol
            # Measure "complete coverage" here => match length / template atom count
            for match in matches:
                ratio = len(match) / float(tpl_n_atoms)
                if ratio > best_ratio:
                    best_ratio = ratio

        per_tpl_scores.append(best_ratio)

    if not per_tpl_scores:
        return 0.0

    # Take minimum: every template fragment must match relatively completely
    return min(per_tpl_scores)


def check_template_match(
    query_reactants_mols,
    query_products_mols,
    template_smarts,
    allow_reverse=False,
    verbose=True,
    score_threshold=0.7,
):
    """
    [NEW VERSION] Option B: Substructure matching (No RunReactants)
    Returns: (is_match: bool, tpl_match_score: float in [0,1])
    - Parse template SMARTS -> ReactionFromSmarts
    - Extract reactant templates
    - Calculate:
        * forward_score  = Template.reactants vs Query.reactants match degree
        * reverse_score  = (Optional) Template.reactants vs Query.products match degree
    - Final match score = max(forward_score, reverse_score)
    - is_match = (score >= score_threshold)
    """
    try:
        rxn = AllChem.ReactionFromSmarts(template_smarts)
    except Exception as e:
        if verbose:
            print(f"[WARN] Unable to parse template SMARTS: {e}")
        return False, 0.0

    tpl_reactants = rxn.GetReactants()
    if len(tpl_reactants) == 0:
        if verbose:
            print("[WARN] No reactant fragments in template, skipping")
        return False, 0.0

    # ---------- Forward: Template.reactants vs Query.reactants ----------
    forward_score = _compute_template_match_score_one_direction(
        query_reactants_mols, tpl_reactants
    )

    # ---------- Reverse (Optional): Template.reactants vs Query.products ----------
    reverse_score = 0.0
    if allow_reverse and query_products_mols:
        reverse_score = _compute_template_match_score_one_direction(
            query_products_mols, tpl_reactants
        )

    score = max(forward_score, reverse_score)

    if verbose:
        print(f"[DEBUG] Template substructure match score = {score:.3f}")

    return (score >= score_threshold), score


# ============================================================
# DEBUG: Print template reactant fragments and their matches in query reaction
# ============================================================
def debug_template_application(query_reactants_mols, query_products_mols, template_smarts):
    """
    No longer RunsReactants, only prints:
    - Each reactant SMARTS in the template
    - Whether it matches substructures in query_reactants / query_products
    """
    print("\n================= DEBUG: TEMPLATE APPLICATION =================")
    print(f"[DEBUG] Template SMARTS: {template_smarts}")

    try:
        rxn = AllChem.ReactionFromSmarts(template_smarts)
    except Exception as e:
        print(f"[ERROR] SMARTS could not be parsed: {e}")
        print("==============================================================\n")
        return

    tpl_reactants = rxn.GetReactants()
    print(f"[DEBUG] Template contains {len(tpl_reactants)} reactant fragments:\n")

    for i, tpl_r in enumerate(tpl_reactants):
        smarts = Chem.MolToSmarts(tpl_r)
        tpl_mol = Chem.MolFromSmarts(smarts)
        print(f"  [Reactant #{i+1}] SMARTS = {smarts}")

        if tpl_mol is None:
            print("    - Could not parse as SMARTS molecule, skipping")
            continue

        # Find matches in reactants
        hit_reactants = []
        for j, qmol in enumerate(query_reactants_mols):
            if qmol.HasSubstructMatch(tpl_mol):
                hit_reactants.append(j)
        print(f"    - Matched molecule indices in query REACTANTS: {hit_reactants if hit_reactants else 'None'}")

        # Find matches in products
        hit_products = []
        for j, qmol in enumerate(query_products_mols):
            if qmol.HasSubstructMatch(tpl_mol):
                hit_products.append(j)
        print(f"    - Matched molecule indices in query PRODUCTS: {hit_products if hit_products else 'None'}")

    print("==============================================================\n")


# ============================================================================
# Database Preprocessing (Fingerprint calculation, Template validation)
# ============================================================================
def preprocess_database(db_path):
    print(f"\n[!] Starting database preprocessing: {db_path}")
    t0 = time.time()

    try:
        with open(db_path, "r") as f:
            db_entries = json.load(f)
    except Exception as e:
        print("Database loading failed:", e)
        return None

    processed = []
    skipped = 0

    for i, entry in enumerate(db_entries):
        if not entry.get("ref_rxn"):
            skipped += 1
            continue

        ref_rxn = entry["ref_rxn"][0]
        fp = calculate_reaction_fingerprint(ref_rxn)

        if not fp:
            skipped += 1
            continue

        tpl = entry.get("smarts", "")
        tpl_valid = True
        if tpl:
            tpl_valid = validate_template_against_reference(ref_rxn, tpl, db_index=i)

        processed.append({
            "fingerprint": fp,
            "original_entry": entry,
            "db_index": i,
            "template_valid": tpl_valid,
        })

    t1 = time.time()
    print(f"Preprocessing complete. Total: {len(processed)} records, Skipped: {skipped}. Time: {t1-t0:.2f}s")
    return processed


# ============================================================================
# Query Main Logic (Fingerprint Filtering + Substructure SMARTS Ranking)
# ============================================================================
def recommend_enzymes(
    query_reaction_smiles,
    preprocessed_db,
    top_k,
    similarity_threshold=0.0,
    tpl_score_threshold=0.7,
):
    print("\n==================== Query Started ====================")
    print("Input Reaction:", query_reaction_smiles)

    query_fp = calculate_reaction_fingerprint(query_reaction_smiles)
    if not query_fp:
        print("Could not generate fingerprint for query reaction!")
        return

    query_reactants, query_products = parse_reaction_smiles(query_reaction_smiles)
    if not query_reactants or not query_products:
        print("Could not parse query reaction!")
        return

    # -------- Phase 1: Fingerprint Filtering --------
    print(f"\n--- Phase 1: Fingerprint Similarity Filtering (Top {top_k}) ---")
    all_fps = [item["fingerprint"] for item in preprocessed_db]
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, all_fps)

    scored = [
        (sim, item) for sim, item in zip(sims, preprocessed_db)
        if sim >= similarity_threshold
    ]

    scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored[:top_k]

    if not top_candidates:
        print("No candidates found via fingerprint filtering!")
        return

    print(f"Phase 1 passed: {len(top_candidates)} candidates")

    # -------- Phase 2: Substructure Template Matching --------
    print("\n--- Phase 2: SMARTS Substructure Ranking (Option B) ---")
    high_conf = []
    mid_conf = []

    for sim, cand in top_candidates:
        entry = cand["original_entry"]
        tpl = entry.get("smarts", "")
        tpl_valid = cand.get("template_valid", True)

        is_match = False
        tpl_score = 0.0

        if tpl and tpl_valid:
            # To observe matching details, enable debug:
            # debug_template_application(query_reactants, query_products, tpl)
            is_match, tpl_score = check_template_match(
                query_reactants,
                query_products,
                tpl,
                allow_reverse=False,
                verbose=False,
                score_threshold=tpl_score_threshold,
            )

        result = {
            "ref_enz": entry.get("ref_enz", ["Unknown"]),
            "similarity": sim,
            "tpl_match_score": tpl_score,
            "db_index": cand["db_index"],
            "ref_rxn": entry["ref_rxn"][0],
            "smarts": tpl,
        }

        if is_match:
            high_conf.append(result)
        else:
            mid_conf.append(result)

    # -------- Output Results --------
    print("\n==================== Recommendation Results ====================")

    print(f"\n✓ High Confidence (Template substructure match success, tpl_match_score ≥ {tpl_score_threshold}): {len(high_conf)} items")
    for r in high_conf:
        print(
            f"  - Enzyme: {r['ref_enz']}, "
            f"Sim={r['similarity']:.4f}, "
            f"TplScore={r['tpl_match_score']:.3f}, "
            f"DB={r['db_index']}"
        )

    print(f"\n● Medium Confidence (Only fingerprint similar or template not applicable): {len(mid_conf)} items")
    for r in mid_conf:
        print(
            f"  - Enzyme: {r['ref_enz']}, "
            f"Sim={r['similarity']:.4f}, "
            f"TplScore={r['tpl_match_score']:.3f}, "
            f"DB={r['db_index']}"
        )

    print("\n==================== Query Finished ====================\n")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":

    DATABASE_FILE = "template_library_smarts_recursion.json"
    PICKLE_FILE = "preprocessed_db.pkl"  # [MOD] Avoid mixing with other versions
    TOP_K_FILTER = 10

    db = None

    if os.path.exists(PICKLE_FILE):
        try:
            print(f"Loading from cache {PICKLE_FILE} ...")
            with open(PICKLE_FILE, "rb") as f:
                db = pickle.load(f)
            print("Loaded successfully!")
        except Exception:
            print("Failed to read cache, re-preprocessing.")

    if db is None:
        if not os.path.exists(DATABASE_FILE):
            print("Database file not found, exiting.")
            sys.exit(1)

        db = preprocess_database(DATABASE_FILE)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(db, f)

    # Example query (Amide hydrolysis with mapping)
    query = "CC1=CC=C(S(N(CC=C)CC=C)(=O)=O)C=C1>>CC1=CC=C(S(N2CC=CC2)(=O)=O)C=C1"
    # "[NH2:1][CH2:2][CH2:3][CH2:4][CH2:5][C:6](N)=[O:7]>>[NH2:1][CH2:2][CH2:3][CH2:4][CH2:5][C:6](=[O:7])[OH:8]"
    recommend_enzymes(query, db, TOP_K_FILTER)
