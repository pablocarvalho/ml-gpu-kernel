#Execution model for rtx 2080 data with metrics output
./stratified_classification_methods.py -i data/rtx-2080_attempts.csv -v -m attempts -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/rtx-2080_concurrency_verbose_metrics.txt

./stratified_classification_methods.py -i data/rtx-2080_CE.csv -v -m ce -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/rtx-2080_interference_verbose_metrics.txt

#Execution model for tesla p100 data with metrics output
./stratified_classification_methods.py -i data/tesla_p100_attempts.csv -v -m attempts -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/p100_concurrency_verbose_metrics.txt

./stratified_classification_methods.py -i data/tesla_p100_CE.csv -v -m ce -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/p100_interference_verbose_metrics.txt


#Execution model for rtx 2080 data with metrics output and grid-search
./stratified_classification_methods.py -i data/rtx-2080_attempts.csv -v -g -m attempts -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/rtx-2080_concurrency_verbose_metrics_grid.txt

./stratified_classification_methods.py -i data/rtx-2080_CE.csv -v -g -m ce -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/rtx-2080_interference_verbose_metrics_grid.txt

# #Execution model for tesla p100 data with metrics output and grid-search
./stratified_classification_methods.py -i data/tesla_p100_attempts.csv -v -g -m attempts -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B  \
2>&1 | tee output/p100_concurrency_verbose_metrics_grid.txt

./stratified_classification_methods.py -i data/tesla_p100_CE.csv -v -g -m ce -p "output/images" \
-o "output/folds" -c k1_blocks_per_grid k2_blocks_per_grid k1_threads_per_block k2_threads_per_block k1_shared_mem_B k2_shared_mem_B \
2>&1 | tee output/p100_interference_verbose_metrics_grid.txt


