# KVSAgg

## Federated Learning

Codes in this folder are forked from open source code of *MinMax Sampling: A Near-optimal Global Summary for Aggregation in the Wide Area*. We apply our KVSAgg after the sampling process in a GPU friendly way.

In cifar10.sh we provided an example of how to run the program. The meaning of some of the arguments are provided in the chart below: 

| Argument name      | meaning                                                               |
|--------------------|-----------------------------------------------------------------------|
| dataset_dir        | directory to where you store the dataset                              |
| num_epochs         | total number of epochs                                                |
| num_clients        | total number of clients                                               |
| num_workers        | the number of clients participate each turn                           |
| num_devices        | the number of gpus                                                    |
| num_rows, num_cols | in FetchSGD, it is used to set the sketch size to num_rows * num_cols |
| k                  | sample size in local topk and our algorithm                           |
| typ                | 0 for HyperIBLT; 1,2,3,4 for CS(1,3,10,0.5); 5,6,7,8 for CSK(1,3,10,0.5)                                                                                  |
| num_buckets        | the width of HyperIBLT                                                |


For FEMNIST, because that we have the orignal data preprocessed into 3597 clients, so the num_clients is always equal to 3597. We run for only 1 epoch for FEMNIST.


## HyperIBLT

We implement HyperIBLT in C++ for CPU platforms and provide an example of computing the encoding and decoding efficiency of HyperIBLT.