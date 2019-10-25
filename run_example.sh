#Execution model for rtx 2080 data with metrics output
./stratified_classification_methods.py -i data/rtx-2080_attempts.csv -v -m attempts 2>&1 | tee data/rtx-2080_concurrency_verbose_metrics.txt
./stratified_classification_methods.py -i data/rtx-2080_CE.csv -v -m ce 2>&1 | tee data/rtx-2080_interference_verbose_metrics.txt

#Execution model for tesla p100 data with metrics output
./stratified_classification_methods.py -i data/tesla_p100_attempts.csv -v -m attempts 2>&1 | tee data/p100_concurrency_verbose_metrics.txt
./stratified_classification_methods.py -i data/tesla_p100_CE.csv -v -m ce 2>&1 | tee data/p100_interference_verbose_metrics.txt