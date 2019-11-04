### Prerequisites

- Python 2.7 with Scikit-Learn (Anaconda Distribution is highly suggested)
  - https://www.anaconda.com/distribution/

- XGboost python libraries ( at least version 0.82 to avoid bugs  )
  - follow the steps in https://pypi.org/project/xgboost/

- funcsigs library
  - follow the steps in https://pypi.org/project/funcsigs/

- sqlite3
  - for a friendly experience when consulting the databases install DB Browser for SQLite   
    https://sqlitebrowser.org/
  

### Repository Structure

- stratified_classification_methods.py :  machine learning script
- run_example.sh :  execution examples used on the paper
- Data : 
  - output files with accuracy, precision, recall and kappa from all experiments.
  - images : precision-recall graphics generated by run_example.sh
  - folds : contains files with classification for each kernel pair used in the experiments, separed by hardware/experiment realized(attempts or ce)/variables used
