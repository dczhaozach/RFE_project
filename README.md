# RFE_project
 Regulation and Firm Dynamism

This project replicates the results of the government regulation restrictions on firm dynamism.

The project is organized as follows:
```console
ML_project
├───data                | store data 
│   ├───cleaned         | final data file
│   ├───processed       | intermediate data file
│   └───raw             | raw data
├───notebooks           | notebooks for exploration/results
├───Src                 | source codes as package
├───results             | graphic and numerical results
├───scripts             | contain main file to execute the code
└───quant_model         | model to explain the results (in progress)
```

The file "data" will not be committed to Github.

To set up the project, first install and run "pipenv" to start the virtual environment in command prompt:
```console
python -m install pipenv
pipenv shell
```

Then install the required package by run
``` console
pipenv install
```

If you are using VS Code, remember to switch to the virtual environment by
``` console
pipenv --venv
```
Then add the displayed path manually to the python interpreter path.

To run the code, simply run the main_file.py in scripts folder.