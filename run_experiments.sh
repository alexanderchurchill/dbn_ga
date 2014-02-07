

for pop_size in 10000 5000 1000
do
	for num_epochs in 25 50 100 200
	do
		for lr in 0.001 0.01 0.1
		do
			nohup python ga.py $pop_size 20 20 $num_epochs $lr 0 &
		done
	done
done