import pandas as pd
import numpy as np
import random
import re
import os
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class QuinazolinoneGenerator:
    """
    Quinazolinone compound generator
    """
    
    def __init__(self, smiles_list):
        """
        Initialize generator
        
        Parameters:
            smiles_list: list of known compound SMILES
        """
        self.smiles_list = smiles_list
        self.ngram_models = {}
        self.scaffolds = []
        self.substituents = []
        self.core_structures = []
        self.linkers = []
        self.end_groups = []
        
        self._build_models()
        self._build_advanced_models()
    
    def _build_models(self, n=3):
        """
        Build N‑gram model
        
        Parameters:
            n: n value for N‑gram, default 3
        """
        print("Building N‑gram model...")
        
        for smi in self.smiles_list:
            padded = '^' * (n-1) + smi + '$' * (n-1)
            for i in range(len(padded) - n + 1):
                ngram = padded[i:i+n]
                prefix = ngram[:-1]
                next_char = ngram[-1]
                if prefix not in self.ngram_models:
                    self.ngram_models[prefix] = Counter()
                self.ngram_models[prefix][next_char] += 1
        
        self._extract_scaffolds()
        self._extract_substituents()
        
        print(f"N‑gram model built, {len(self.ngram_models)} prefixes total")
        print(f"{len(self.scaffolds)} scaffold variants extracted")
        print(f"{len(self.substituents)} substituent types extracted")
    
    def _extract_scaffolds(self):
        """Extract quinazolinone scaffold variants"""
        scaffold_patterns = [
            'O=c1[nH]c2ccccc2c(=O)[nH]1',
            'O=c1nc2ccccc2c(=O)[nH]1',
            'O=C1C=2C(NC=N1)=CC=CC2',
            'O=c1c2ccccc2ncn1',
            'c1ccc2c(=O)[nH]cnc2c1',
        ]
        
        for smi in self.smiles_list[:2000]:
            if 'c(=O)' in smi and ('n1' in smi or 'n2' in smi):
                core_match = re.search(r'c\d*\(=O\)[^)]*n\d*', smi)
                if core_match:
                    core = core_match.group(0)
                    if len(core) > 5 and len(core) < 50:
                        self.scaffolds.append(core)
        
        self.scaffolds = list(set(self.scaffolds))
        if len(self.scaffolds) < 10:
            self.scaffolds = scaffold_patterns
    
    def _extract_substituents(self):
        """Extract common substituents"""
        substituent_patterns = [
            'c1ccccc1',
            'c1ccc(Cl)cc1',
            'c1ccc(F)cc1',
            'c1ccc(OC)cc1',
            'CC',
            'CCC',
            'c1ccco1',
            'c1cccs1',
            'C(=O)N',
            'S(=O)(=O)',
            'N(C)C',
            'c1ccc([N+](=O)[O-])cc1',
            'c1ccc(C)cc1',
            'C(F)(F)F',
            'CN',
            'CCN',
        ]
        
        extracted = []
        for smi in self.smiles_list[:3000]:
            subs = re.findall(r'\(([^)]+)\)', smi)
            extracted.extend([s for s in subs if len(s) > 2])
        
        sub_counts = Counter(extracted)
        self.substituents = [sub for sub, count in sub_counts.most_common(100) if count >= 3]
        
        self.substituents.extend(substituent_patterns)
        self.substituents = list(set(self.substituents))
    
    def _build_advanced_models(self):
        """Build advanced generation models"""
        print("Building advanced generation models...")
        
        self.core_structures = []
        for smi in self.smiles_list:
            matches = re.findall(r'c\d*\(=O\)[^)]*[nN]\d*[^)]*c\d*\(=O\)', smi)
            if matches:
                self.core_structures.extend(matches)
        
        if not self.core_structures:
            self.core_structures = [
                'O=c1[nH]c2ccccc2c(=O)[nH]1',
                'O=c1nc2ccccc2c(=O)[nH]1',
                'c1ccc2c(=O)[nH]cnc2c1',
            ]
        
        self.linkers = ['', 'C', 'CC', 'CCC', 'CCCC', 'C(=O)', 'C(=O)N', 'NC(=O)', 
                       'S', 'O', 'N', 'CC(=O)', 'CCN', 'CNC']
        
        self.end_groups = []
        end_patterns = [
            r'c\d*cc\w*cc\d*',
            r'[CN]\d*\([^)]*\)',
            r'C\(=O\)[^)]*',
        ]
        for smi in self.smiles_list[:5000]:
            for pattern in end_patterns:
                matches = re.findall(pattern, smi)
                self.end_groups.extend(matches)
        
        self.end_groups = list(set(self.end_groups))
        if len(self.end_groups) < 50:
            self.end_groups.extend([
                'c1ccccc1', 'c1ccc(F)cc1', 'c1ccc(Cl)cc1', 'c1ccc(OC)cc1',
                'c1ccco1', 'c1cccs1', 'C(F)(F)F', 'CC', 'CCC', 'N(C)C',
                'N1CCOCC1', 'N1CCCCC1', 'c1ccc([N+](=O)[O-])cc1',
            ])
        
        print(f"Core structures: {len(self.core_structures)}")
        print(f"Linkers: {len(self.linkers)}")
        print(f"End groups: {len(self.end_groups)}")
    
    def generate_smiles(self, n=13000):
        """
        Generate new SMILES (basic version)
        
        Parameters:
            n: number of compounds to generate
        Returns:
            list of generated SMILES
        """
        generated = []
        max_attempts = n * 3
        attempts = 0
        
        print(f"Start generating {n} new compounds...")
        
        while len(generated) < n and attempts < max_attempts:
            attempts += 1
            
            if random.random() < 0.6:
                new_smi = self._ngram_generate()
            elif random.random() < 0.75:
                new_smi = self._scaffold_substituent_generate()
            else:
                new_smi = self._mutate_existing()
            
            if self._is_valid_quinazolinone(new_smi) and new_smi not in generated:
                generated.append(new_smi)
                
            if len(generated) % 2000 == 0:
                print(f"  Generated {len(generated)}/{n} compounds...")
        
        print(f"Generation complete! {len(generated)} new compounds generated")
        return generated
    
    def generate_optimized(self, n=13000):
        """
        Optimized generation method (recommended)
        
        Parameters:
            n: number of compounds to generate
        Returns:
            list of generated SMILES
        """
        generated = []
        seen = set()
        
        print(f"Start optimized generation for {n} compounds...")
        
        for i in range(n * 5):
            if len(generated) >= n:
                break
            
            strategy = random.random()
            
            if strategy < 0.4:
                new_smi = self._core_based_generate()
            elif strategy < 0.7:
                new_smi = self._hybrid_generate()
            elif strategy < 0.9:
                new_smi = self._modify_generate()
            else:
                new_smi = self._ngram_generate()
            
            if new_smi and new_smi not in seen and self._is_valid_quinazolinone(new_smi):
                generated.append(new_smi)
                seen.add(new_smi)
                
                if len(generated) % 2000 == 0:
                    print(f"  Generated {len(generated)}/{n} compounds...")
        
        return generated
    
    def _ngram_generate(self, max_len=150):
        """
        N‑gram based SMILES generation
        
        Parameters:
            max_len: maximum SMILES length
        Returns:
            generated SMILES string
        """
        n = 3
        result = '^' * (n-1)
        
        for _ in range(max_len):
            prefix = result[-(n-1):]
            if prefix not in self.ngram_models:
                break
            
            candidates = self.ngram_models[prefix]
            total = sum(candidates.values())
            r = random.randint(1, total)
            cumulative = 0
            next_char = '$'
            for char, count in candidates.items():
                cumulative += count
                if r <= cumulative:
                    next_char = char
                    break
            
            if next_char == '$':
                break
            result += next_char
        
        smiles = result.replace('^', '').replace('$', '')
        return smiles
    
    def _scaffold_substituent_generate(self):
        """Scaffold + substituent recombination generation"""
        scaffold = random.choice(self.scaffolds)
        
        n_subs = random.randint(1, 3)
        result = scaffold
        
        for _ in range(n_subs):
            sub = random.choice(self.substituents)
            if random.random() < 0.5:
                result = sub + result
            else:
                result = result + sub
        
        return result
    
    def _mutate_existing(self):
        """Mutate existing SMILES"""
        base = random.choice(self.smiles_list)
        
        mutations = [
            ('C', 'N'), ('N', 'C'), ('C', 'O'), ('O', 'C'),
            ('F', 'Cl'), ('Cl', 'F'), ('H', 'F'), ('F', 'H'),
            ('c1ccccc1', 'c1ccc(F)cc1'), ('c1ccc(F)cc1', 'c1ccccc1'),
        ]
        
        old, new = random.choice(mutations)
        mutated = base.replace(old, new, 1)
        
        return mutated
    
    def _core_based_generate(self):
        """Core‑structure based generation"""
        core = random.choice(self.core_structures)
        
        n_subs = random.randint(1, 2)
        result = core
        
        for _ in range(n_subs):
            linker = random.choice(self.linkers)
            end = random.choice(self.end_groups)
            
            if random.random() < 0.5:
                if linker:
                    result = end + linker + result
                else:
                    result = end + result
            else:
                if linker:
                    result = result + linker + end
                else:
                    result = result + end
        
        return result
    
    def _hybrid_generate(self):
        """Hybrid two molecules"""
        smi1 = random.choice(self.smiles_list)
        smi2 = random.choice(self.smiles_list)
        
        parts1 = re.split(r'(C\(=O\)|NC|CC|CCC)', smi1)
        parts2 = re.split(r'(C\(=O\)|NC|CC|CCC)', smi2)
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            if random.random() < 0.5:
                new_smi = parts1[0] + random.choice(parts2[1:])
            else:
                new_smi = random.choice(parts1[:-1]) + parts2[-1]
        else:
            mid1 = len(smi1) // 2
            mid2 = len(smi2) // 2
            new_smi = smi1[:mid1] + smi2[mid2:]
        
        return new_smi
    
    def _modify_generate(self):
        """Modify existing molecule"""
        base = random.choice(self.smiles_list)
        
        modifications = [
            ('F', 'Cl'), ('Cl', 'F'), ('F', 'Br'), ('Br', 'F'),
            ('c1ccccc1', 'c1ccc(F)cc1'), ('c1ccc(F)cc1', 'c1ccccc1'),
            ('c1ccccc1', 'c1ccc(Cl)cc1'), ('c1ccc(Cl)cc1', 'c1ccccc1'),
            ('C', 'N'), ('N', 'C'), ('O', 'S'), ('S', 'O'),
            ('CC', 'CCC'), ('CCC', 'CC'), ('CC', 'CCCC'),
        ]
        
        new_smi = base
        for _ in range(random.randint(1, 3)):
            old, new = random.choice(modifications)
            if old in new_smi:
                new_smi = new_smi.replace(old, new, 1)
        
        return new_smi
    
    def _is_valid_quinazolinone(self, smiles):
        """
        Validate valid quinazolinone structure
        
        Parameters:
            smiles: SMILES string
        Returns:
            boolean validity
        """
        if not smiles or len(smiles) < 10 or len(smiles) > 200:
            return False
        
        has_carbonyl = 'c(=O)' in smiles or 'C(=O)' in smiles
        has_nitrogen = 'n' in smiles.lower()
        has_aromatic = 'c' in smiles.lower()
        
        quinazolinone_features = [
            'c(=O)', 'c2ccccc2', 'n1', 'n2', 'nc', 'c(=O)n',
        ]
        feature_count = sum(1 for f in quinazolinone_features if f in smiles)
        
        brackets_balanced = smiles.count('(') == smiles.count(')')
        
        return has_carbonyl and has_nitrogen and has_aromatic and feature_count >= 2 and brackets_balanced


def calculate_smiles_similarity(smiles1, smiles2, n=3):
    """
    Calculate SMILES similarity based on common substrings
    
    Parameters:
        smiles1, smiles2: two SMILES strings
        n: N‑gram size
    Returns:
        similarity score (0‑1)
    """
    def get_ngrams(s, n):
        return set([s[i:i+n] for i in range(len(s)-n+1)])
    
    ngrams1 = get_ngrams(smiles1, n)
    ngrams2 = get_ngrams(smiles2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def simple_cluster(smiles_list, n_clusters=10):
    """
    Simple clustering based on similarity
    
    Parameters:
        smiles_list: list of SMILES
        n_clusters: number of clusters
    Returns:
        clusters: cluster result list
        centers: cluster centers
    """
    clusters = [[] for _ in range(n_clusters)]
    
    centers = random.sample(smiles_list, n_clusters)
    
    for smi in smiles_list:
        similarities = [calculate_smiles_similarity(smi, center) for center in centers]
        best_cluster = similarities.index(max(similarities))
        clusters[best_cluster].append(smi)
    
    return clusters, centers


def estimate_mw(smiles):
    """
    Estimate molecular weight (simplified)
    
    Parameters:
        smiles: SMILES string
    Returns:
        estimated molecular weight
    """
    mw = 0
    mw += smiles.count('C') * 12
    mw += smiles.count('c') * 12
    mw += smiles.count('N') * 14
    mw += smiles.count('n') * 14
    mw += smiles.count('O') * 16
    mw += smiles.count('o') * 16
    mw += smiles.count('S') * 32
    mw += smiles.count('F') * 19
    mw += smiles.count('Cl') * 35.5
    mw += smiles.count('Br') * 80
    mw += smiles.count('I') * 127
    mw += (smiles.count('C') + smiles.count('c')) * 1.5
    return mw


def calculate_priority_score(smi, generated_smiles_opt):
    """
    Calculate molecule priority score
    
    Parameters:
        smi: SMILES string
        generated_smiles_opt: list of all generated molecules
    Returns:
        priority score (0‑1)
    """
    score = 0
    
    sample_check = random.sample(generated_smiles_opt, min(20, len(generated_smiles_opt)))
    avg_sim = np.mean([calculate_smiles_similarity(smi, s) for s in sample_check])
    diversity_score = 1 - avg_sim
    
    complexity = smi.count('(') + smi.count('=') + smi.count('[')
    complexity_score = min(complexity / 20, 1.0)
    
    drug_like_score = 0
    if 'c1ccccc1' in smi: drug_like_score += 0.2
    if 'C(=O)N' in smi: drug_like_score += 0.2
    if 'N' in smi: drug_like_score += 0.1
    if all(x not in smi for x in ['[N+](=O)[O-]', 'S(=O)(=O)']): drug_like_score += 0.2
    if 10 < len(smi) < 150: drug_like_score += 0.3
    
    total_score = diversity_score * 0.4 + complexity_score * 0.3 + drug_like_score * 0.3
    return total_score


def main():
    """
    Main function - full generation pipeline
    """
    print("="*80)
    print("Quinazolinone Compound Generator")
    print("="*80)
    
    print("\n1. Load raw data...")
    input_file = '/mnt/kimi/upload/Akzlt.xlsx'
    df = pd.read_excel(input_file)
    smiles_list = df.iloc[:, 0].dropna().tolist()
    print(f"   Loaded {len(smiles_list):,} compounds")
    
    print("\n2. Initialize generator...")
    generator = QuinazolinoneGenerator(smiles_list)
    
    print("\n3. Generate new compounds...")
    generated_smiles = generator.generate_optimized(n=13000)
    
    print("\n4. Build result DataFrame...")
    result_df = pd.DataFrame({
        'smiles': generated_smiles,
        'compound_id': [f'GEN_{i+1:05d}' for i in range(len(generated_smiles))],
        'smiles_length': [len(s) for s in generated_smiles]
    })
    
    print("\n5. Calculate priority scores...")
    priority_scores = [calculate_priority_score(smi, generated_smiles) 
                      for smi in generated_smiles]
    result_df['priority_score'] = priority_scores
    result_df['priority_rank'] = result_df['priority_score'].rank(ascending=False).astype(int)
    result_df['priority_class'] = pd.cut(result_df['priority_score'], 
                                         bins=[0, 0.4, 0.6, 0.8, 1.0],
                                         labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("\n6. Analyze similarity and diversity...")
    sample_size = 500
    original_sample = random.sample(smiles_list, sample_size)
    generated_sample = random.sample(generated_smiles, sample_size)
    
    similarities = []
    for gen_smi in generated_sample[:100]:
        max_sim = max([calculate_smiles_similarity(gen_smi, orig_smi) 
                      for orig_smi in original_sample[:200]])
        similarities.append(max_sim)
    
    internal_sims = []
    for i in range(50):
        smi1, smi2 = random.sample(generated_smiles, 2)
        internal_sims.append(calculate_smiles_similarity(smi1, smi2))
    
    print("\n7. Count chemical features...")
    substituent_types = {
        'Phenyl': sum(1 for s in generated_smiles if 'c1ccccc1' in s),
        'Halogen': sum(1 for s in generated_smiles if any(h in s for h in ['F', 'Cl', 'Br'])),
        'Nitro': sum(1 for s in generated_smiles if '[N+](=O)[O-]' in s),
        'Methoxy': sum(1 for s in generated_smiles if 'OC' in s),
        'Amide': sum(1 for s in generated_smiles if 'C(=O)N' in s),
        'Sulfonyl': sum(1 for s in generated_smiles if 'S(=O)(=O)' in s),
    }
    
    print("\n8. Save output files...")
    output_dir = '/mnt/kimi/output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/Quinazolinone_Library.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='All_Compounds', index=False)
        
        top_compounds = result_df[result_df['priority_class'].isin(['High', 'Very High'])].head(1000)
        top_compounds.to_excel(writer, sheet_name='Top_1000_Priority', index=False)
        
        stats_df = pd.DataFrame({
            'Metric': ['Total', 'Avg Length', 'Avg Priority', 'Diversity'],
            'Value': [
                len(result_df),
                result_df['smiles_length'].mean(),
                result_df['priority_score'].mean(),
                1 - np.mean(internal_sims)
            ]
        })
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print(f"\n{'='*80}")
    print("Generation finished!")
    print(f"Total compounds: {len(generated_smiles):,}")
    print(f"Output file: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
