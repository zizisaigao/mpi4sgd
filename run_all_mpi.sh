#!/bin/bash

# output file
LOG_FILE="mpi_train_log.txt"
RESULT_CSV="results.csv"
HOSTFILE="./hostfile"

# clear or create log and CSV files.
> "$LOG_FILE"
echo "activ,batch,hidden,procs,train_rmse,test_rmse,time_sec" > "$RESULT_CSV"

# hyperparameter
ACTIVATIONS=("sigmoid" "relu" "tanh")
BATCH_SIZES=(128 256 512 1024 2048)
HIDDEN_SIZES=(32)
PROCS_TO_RUN=(4) # Set the number of processes according to requirements
HOSTFILE="./hostfile"

# lr and batch
BASE_LR=0.01
BASE_BATCH=128

# Epoch
EPOCH_COUNT=20

# print start info
echo "=== MPI Training Experiment Started ===" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Activation functions: ${ACTIVATIONS[*]}" | tee -a "$LOG_FILE"
echo "Batch sizes: ${BATCH_SIZES[*]}" | tee -a "$LOG_FILE"
echo "Hidden sizes: ${HIDDEN_SIZES[*]}" | tee -a "$LOG_FILE"
echo "Number of processes: ${PROCS_TO_RUN[*]}" | tee -a "$LOG_FILE"
echo "Number of epochs: $EPOCH_COUNT" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# iterate over all combinations: activ × batch × hidden × procs
for activ in "${ACTIVATIONS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        for hidden in "${HIDDEN_SIZES[@]}"; do
            # calculate lr
            lr=$(awk "BEGIN {print ${BASE_LR} * ${batch} / ${BASE_BATCH}}")
            for procs in $PROCS_TO_RUN; do
                echo "" | tee -a "$LOG_FILE"
                echo ">>> running: activ=$activ, batch=$batch, hidden=$hidden, procs=$procs" | tee -a "$LOG_FILE"
                echo "Start time: $(date)" | tee -a "$LOG_FILE"
                echo "learning rate: $lr" | tee -a "$LOG_FILE"
                echo "----------------------------------------" | tee -a "$LOG_FILE"

                # execute the command and redirect both stdout and stderr to a temporary log file
                TMP_LOG=$(mktemp)
                mpirun --hostfile "$HOSTFILE" -n $procs python nn_mpi.py --data_dir "data" --hidden $hidden --batch $batch --activ $activ --lr $lr --epoch $EPOCH_COUNT | tee "$TMP_LOG"

                # check
                if [ $? -eq 0 ]; then
                    echo "✅ Successfully completed" | tee -a "$LOG_FILE"
                else
                    echo "❌ Command failed. Please check" | tee -a "$LOG_FILE"
                fi

                # tmp log to main file
                cat "$TMP_LOG" >> "$LOG_FILE"

                # parse the log and extract the results.
                train_rmse=$(grep -E '\[RESULT\] Train RMSE\s*=\s*[-+0-9.eE]+' "$TMP_LOG" | awk -F'=' '{print $2}' | xargs)
                test_rmse=$(grep -E '\[RESULT\] Test RMSE\s*=\s*[-+0-9.eE]+' "$TMP_LOG" | awk -F'=' '{print $2}' | xargs)
                time_sec=$(grep -E '\[INFO\] Training time\s*[0-9.]+' "$TMP_LOG" | awk '{print $4}')

                # avoid null
                train_rmse=${train_rmse:-nan}
                test_rmse=${test_rmse:-nan}
                time_sec=${time_sec:-nan}

                # write into csv
                echo "$activ,$batch,$hidden,$procs,$train_rmse,$test_rmse,$time_sec" >> "$RESULT_CSV"

                # clear tmp log
                rm "$TMP_LOG"
            done
        done
    done
done

echo "========================================" | tee -a "$LOG_FILE"
echo "=== All experiments completed. Results have been saved to $RESULT_CSV ===" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"