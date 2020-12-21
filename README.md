# mllib

```
git clone https://github.com/eltoto1219/mllib.git && cd mllib && pip install -e .
```
once data ready, all one needs to do is provide experiment, and dataset names, ie.
```
run exp evallxmert gqa --yaml=libdata/eval_test.yaml --save_on_crash=true --email=your@email.com
```
some immidiately visible funcitonality:
1. load config from yaml (can dump config to yaml to) 
2. save experiment information, models, optimizers, etc .. on crash for perfect recovery
3. send email on experiment crash
<br />
currently `run extract` is broken <br />

TODO:<br />
1. mv processors/general.py to processors/__main__.py
2. generate experiment_dict, loop_dict, models_dict, proccessors_dict, and datasets_dict dynamically instead of hardcoding
3. mount public google drive directory and upload all arrow and other data
4. create download.py for auto downloading of data/files into respective dirs set in config if not present
4.a. dowload from official dataset websites + mounted google drive filesystem
