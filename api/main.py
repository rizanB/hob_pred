import json
import os

import numpy as np
import pubchempy as pcp
from fastapi import FastAPI
from rdkit.Chem import Descriptors, MolFromSmiles
from utils.load_conf import load_config

app = FastAPI()
config = load_config()

BASE_DIR = config['paths']['BASE_DIR']

av_models_path = os.path.join(BASE_DIR, "reports", "av_models.json")
with open(av_models_path, "r") as f:
    av_models = json.load(f)

@app.post("/predict_hob")
def predict_hob(cmp_name:str, model_name:str = "knn"):

    cmp_name = cmp_name.lower()
    model_name = model_name.lower()

    if cmp_name == "":
        return f"compound name cannot be empty!"

    # fetch descriptors
    r = fetch_drug_features(cmp_name)

    if r == 0:
        return f"compound not found in pubchem"
    else:
        f = r

    mpath = get_mpath(model_name)

    # todo: scale feature set and predict hob

    return(f"{f, mpath}")


# takes drug name, returns feature set, if no cmp found: returns 0
def fetch_drug_features(_cmp_name):
    c = pcp.get_compounds(_cmp_name, "name")

    if not c:
        return 0

    smiles = c[0].canonical_smiles
    mol = MolFromSmiles(smiles)

    f = []
    hacc = Descriptors.NumHAcceptors(mol)
    hdon = Descriptors.NumHDonors(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    tpsa = Descriptors.TPSA(mol)
    nrot = Descriptors.NumRotatableBonds(mol)

    f.append(hacc)
    f.append(hdon)
    f.append(mw)
    f.append(logp)
    f.append(mr)
    f.append(tpsa)
    f.append(nrot)

    # todo: fetch acid/base category of mol
    return f


# get model path based on model picked
def get_mpath(_model_name):

# pick model: if user specified invalid names, pick knn instead
    if not (_model_name in av_models.keys()):
        _model_name = "knn"

    mname_acc = f"{_model_name}_{av_models[_model_name][1]}"
    mpath = os.path.join(BASE_DIR, "models", f"{mname_acc}.joblib")
    return mpath
