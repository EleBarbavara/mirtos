## Install mirtos

You can install the repo by cloning it from the terminal: 
```bash
git clone url_repo
```

## Environment setup

After cloning the repo, you have to setup the environment. Mirtos requires a python version >=3.12 . To setup the virtual enviroment, use poetry. In the terminal:
```bash
curl -sSL https://install.python-poetry.org | python3.14 -
# Let the system know where poetry is installed
echo 'export PATH="$HOME/Library/Application Support/pypoetry/venv/bin:$PATH"' >> ~/.zshrc
# apply the changes with
source .zshrc
```
Then, in the mirtos folder, a poetry.toml file is present. In this folder:
```bash
poetry install
```
To check the environment information:
```bash
poetry env info
```

To activate the virtual environment, check if a .venv folder is present in the mirtos folder and you have a compatible python version:
```bash
ls -la
python --version  #should be >=3.12
```
and then:
```bash
source .venv/bin/activate
```
If the environment is correctly activated, the terminal line should appear as:
```bash
(mirtos-py3.14) (base) eleonora@dhwired13085 mirtos % 
```

## Run a script
To run a basic version of mirtos, go in the mapmaking folder and run the following line:
```bash
cd src/mirtos/mapmaking
python mapmaking.py path/to/config.yaml
```
**WARNING**: if you are a user and editor of this repo, we advise to not put your config file in the configs folder, but to store it locally on you computer. The same applies to data used to make maps. 


