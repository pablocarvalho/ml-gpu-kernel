### Prerequisites

- Python 2.7 (Anaconda Distribution is highly suggested)
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
- run_example.sh :  input parameters example for the script
- Data : 
  - kernels-P100.db : sqlite database containing nvprof raw kernel data outputs for tesla P100 experiments
  - kernels-rtx2080.db :  sqlite database containing nvprof raw kernel data outputs for rtx 2080 experiments	
  - querry.sql :  Querries used to create input files for machine learning script
  - rtx-2080_CE.csv : interference input file for machine learning script for RTX 2080 
  - rtx-2080_attempts.csv : concurrency input file for machine learning script for RTX 2080    
  - tesla_p100_CE.csv : interference input file for machine learning script for P100
  - tesla_p100_attempts.csv : concurrency input file for machine learning script for P100
  - rtx-2080_concurrency_verbose_metrics.txt :  terminal output file with detailed metrics for each folds and its averages
  - rtx-2080_interference_verbose_metrics.txt : terminal output file with detailed metrics for each folds and its averages   
  - p100_concurrency_verbose_metrics.txt :  terminal output file with detailed metrics for each folds and its averages
  - p100_interference_verbose_metrics.txt : terminal output file with detailed metrics for each folds and its averages
