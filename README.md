# mllib
to install (add editable for personal custimization)
```
git clone https://github.com/eltoto1219/mllib.git && cd mllib && pip install -e .
```
once data ready, all one needs to do is provide experiment, and dataset names, ie.
```
run exp lxmert gqa --yaml=libdata/eval_test.yaml --save_on_crash=true --email=your@email.com
```
some immidiately visible funcitonality:
1. load config from yaml (can dump config to yaml to) 
2. save experiment information, models, optimizers, etc .. on crash for perfect recovery
3. send email on experiment crash


<br />
The idea is to make research prototyping/development as fast as possible. All the while, keeping all of your work confidential.
We can do this by specifying a "private" experiment file and add a "private_file=sample.py" flag or option in a yaml file ("sample.yaml")

```
run exp sample gqa --yaml=sample.yaml --save_on_crash=true --email=your@email.com
```

if we, instead run the following command:

```
run exp sample gqa --save_on_crash=true --email=your@email.com
```

and do NOT include the "sample.yaml" flag which points to the private file, "sample.py", the the `run exp` command will throw an error as the experiment "sample" is not registerd into this repo by default.
<br />
<br />

!!!! <br />
currently `run extract` is broken <br />
!!!!!<br />



TODO:<br />
3. mount public google drive directory and upload all arrow and other data
4. create download.py for auto downloading of data/files into respective dirs set in config if not present
4.a. dowload from official dataset websites + mounted google drive filesystem


<br />
(a much better documentation website is underway)
