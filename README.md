# mllib

```
git clone https://github.com/eltoto1219/mllib.git && cd mllib && pip install -e .
```

currently `run extract` is broken
TODO:
1. mv processors/general.py to processors/__main__.py
2. generate experiment_dict, loop_dict, models_dict, proccessors_dict, and datasets_dict dynamically instead of hardcoding
3. mount public google drive directory and upload all arrow and other data
4. create download.py for auto downloading of data/files into respective dirs set in config if not present
4.a. dowload from official dataset websites + mounted google drive filesystem
