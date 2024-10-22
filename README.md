# SkyExp
GitHub Repo for SkyExp.

## 1. Install packages using requirements.txt
```
pip install -r requirements.txt
```

## 2. Train the model. (Some models are provided.)
```
./train.sh config.yaml train_results/
```

## 3. Run the experiments. (IPF score)
```
./run.sh config.yaml results/
```

## 4. The edge masks of explanations are stored in src.results, together with baseline results, the IGD and MS scores can be calculated. 
