
	
python3 cv_train.py\
		--mode local_topk \
		--error_type local \
		--num_clients 200 \
		--num_workers 12 \
		--num_epochs 50 \
		--pivot_epoch 10 \
		--k 218955 \
        --numIBLT 300000 \
        --typ 1 \
		--local_batch_size 50 --dataset_dir ~/cifar10 --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --num_devices 1 --num_rows 1 --share_ps_gpu --outlier_thres 100


python3 cv_train.py\
		--mode sampling \
		--num_clients 2000 \
		--num_workers 20 \
		--num_epochs 30 \
		--pivot_epoch 6 \
		--k 218955 \
        --numIBLT 300000 \
        --typ 0 \
		--num_cols 1000000 \
		--local_batch_size 25 --dataset_dir ~/cifar10 --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --num_devices 1 --num_rows 1 --share_ps_gpu --outlier_thres 100

