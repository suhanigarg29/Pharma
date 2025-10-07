import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, DataStructs

# load objects
model = joblib.load('models/qsar_xgb_acetylcholinesterase.joblib')
feature_names = joblib.load('models/feature_names.joblib')
pca = joblib.load('models/descriptor_pca.joblib')

nBits = 1024

# helper functions (same as notebook)
def mol_from_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None

desc_columns = [c for c in feature_names if not c.startswith('FP_')]

def compute_descriptors_for_app(mol):
    # same desc names used in the notebook
    desc_names = ['MolLogP','MolWt','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds','FractionCSP3','NumValenceElectrons']
    dvals = []
    for name in desc_names:
        try:
            dvals.append(getattr(Descriptors, name)(mol))
        except:
            dvals.append(np.nan)
    return dvals

def make_feature_vector_from_smiles(smi):
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    # descriptors
    desc_vals = compute_descriptors_for_app(mol)
    # fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    fp_arr = np.array(fp).astype(int)
    # build same-order vector as feature_names
    vec = []
    # descriptors (desc_columns order), but we used fixed desc order — keep consistency
    # Here we assume desc_columns matches the names used above; for safety, map names where possible
    for col in feature_names:
        if col.startswith('FP_'):
            idx = int(col.split('_')[1])
            vec.append(int(fp_arr[idx]))
        else:
            # map descriptor name to index in desc_vals list (based on known desc order)
            # this is a simplification — ensure descriptor names match notebook
            try:
                mapping = {
                    'MolLogP':0,'MolWt':1,'TPSA':2,'NumHDonors':3,'NumHAcceptors':4,'NumRotatableBonds':5,'FractionCSP3':6,'NumValenceElectrons':7
                }
                vec.append(float(desc_vals[mapping[col]]))
            except:
                vec.append(0.0)
    return np.array(vec)

st.title("QSAR — AChE pIC50 predictor")
smi = st.text_input("Enter SMILES")
if st.button("Predict"):
    vec = make_feature_vector_from_smiles(smi)
    if vec is None:
        st.error("Invalid SMILES")
    else:
        pred = model.predict([vec])[0]
        st.success(f"Predicted pIC50: {pred:.2f}")
        mol = Chem.MolFromSmiles(smi)
        st.image(Draw.MolToImage(mol))
