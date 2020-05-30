
# this will grid search parameters that learn mackey-glass
# print out --help parameters for our cmdline script, so you know what you can tune:
fn_mackey_glass --help



# If you wish to start a new runlog.csv file, uncomment this line
echo "timestamp,publog_train_err,publog_test_err,publog_runtime_sec,hidden_size,output_size,seed,num_layers,nonlinearity,batch_first,leaking_rate,spectral_radius,w_io,w_ih_scale,lambda_reg,density,readout_training,output_steps,dataset" > runlog.csv

hiddenSize=100

for i in `seq 0.1 0.2 2.0 `; do

   fn_mackey_glass --hidden_size ${hiddenSize} --num_layers 1 --spectral_radius ${i} >>runlog.csv

   cat runlog.csv | gawk -F, 'BEGIN{OFS=","}{print $1, $2,$3,$12, $5,$8}'| column -t -s ","
   echo ""

done

