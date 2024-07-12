
## Ambiente:
1. usando conda, criado um ambiente python 3.11
2. instalado poetry no ambiente conda
3. instalado pandas, numpy, matplotlib, seaborn, pytorch, optuna, mlflow, jupyter, ipykernel

### Como:

Abra um terminal na pasta onde vai criar o notebook e execute:

``` bash 
conda create -n epd_env python=3.11 -y
```

``` bash 
conda activate epd_env
```

``` bash 
pip install poetry
```

``` bash 
poetry new epd_kit
```


``` bash 
cd epd_kit
```

``` bash 
poetry config virtualenvs.create false --local
```

``` bash 
poetry add torch mlflow optuna seaborn ipykernel tensorboard
```

<!-- poetry add mlflow optuna torch pandas numpy matplotlib seaborn jupyter ipykernel tensorboard -->

Optional:

``` bash 
poetry show
``` 


```base
poetry add pyqt5
```

```base
poetry add mat73
```

```base
poetry add gym
```

```base
poetry add stable_baselines3
```

```base
poetry add stable_baselines3[extra]
```


## Troubleshooting:

Install using cmd in windows

Remove conda env if necessary:
```bash
conda remove --name epd_env --all -y
```


