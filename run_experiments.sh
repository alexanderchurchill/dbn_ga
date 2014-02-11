

for pop_size in 1000 5000 10000
do
	for num_epochs in 50
	do
		for lr in 0.1 0.01 0.001
		do
			for corruption_level in 0.1 0.2 0.3
			do
				nohup python ga.py $pop_size 20 20 $num_epochs $lr 0 $corruption_level &
			done
		done
	done
done