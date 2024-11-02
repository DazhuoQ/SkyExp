# SkyExp
GitHub Repo for SkyExp. All experimental settings are in config.yaml. 

## 1. Install packages using requirements.txt
```
pip install -r requirements.txt
```

## 2. Train your own model. 
```
./train.sh config.yaml train_results/
```

## 3. Run the experiments. (IPF score)
```
./run.sh config.yaml results/
```

The edge masks of explanations are stored in src.results, together with baseline results, the IGD and MS scores can be calculated. 
