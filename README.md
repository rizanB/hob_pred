# Hob_pred: Prediction of human oral bioavailability from drug molecular descriptors

### (work in progress)
predicts oral bioavailability of drugs from their chemical features (like h-bonds, tpsa, molwt, nrot, etc.) using classical models and ensembles.

prepared dataset, trained models, basic endpoints (pending: get prediction)
 
## Running locally
- clone the repo
- install dependencies from requirements.txt
- run ```uvicorn main:app --reload``` within api folder

## Endpoints

### Available models
[GET] ```localhost:8000/available_models```

### Get hob prediction 
[POST] ```localhost:8000/predict_hob?cmp_name=gallic acid*model_name=rf```
