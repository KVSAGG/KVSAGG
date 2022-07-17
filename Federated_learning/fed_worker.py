import torch
import numpy as np
import ctypes
from ctypes import *
from utils import get_param_vec, set_param_vec, get_grad, _topk, _sampling, clip_grad, _nips
import copy
import os
import time
import math
import torch.multiprocessing as multiprocessing
from csvec import CSVec
import torch.distributed as dist
import queue
import resource
import csvec
from secure_aggre import client
from secure_aggre import server
from secure_aggre import client_float
from secure_aggre import server_float
import datetime
import time
torch.set_printoptions(profile="full")
torch.set_printoptions(sci_mode=False)
class data(ctypes.Structure):
    _fields_ = [("k",c_int), ("idx",c_int * 3000010), ("grad", c_double * 3000010)]

dd = 6568640
cc = 300000
rr = 3


numIBLT = 0
numCS = 0
numCSk = 0
CS = 0
CSk = 0
IBLT = 0
IBLT_arr = []

dll2 = ctypes.cdll.LoadLibrary('./c.so')



print("IBLT initialization finished")
class IBL_T(ctypes.Structure):
    _fields_ = [("r",c_int),("c",c_int),("d",c_int),("id1",(c_int*cc)*rr),("fingerprint",(c_int*cc)*rr),("counter",(c_int*cc)*rr),("value",(c_double*cc)*rr),
    ("hash_fing",c_int*dd),("buckets",(c_int*dd)*rr)]

# iblts = (IBL_T*30)()

class data_set(ctypes.Structure):
    _fields_ = [("din",data * 10)]


################################################aggre##############################################################

ze = torch.zeros(rr,cc)
Server = server.secaggserver("127.0.0.1", 2019, 4, 4)
s = []
for i in range(0, 4):
    s.append(client.secaggclient("127.0.0.1",2019))
    s[i].set_weights(ze,(rr,cc))
    s[i].set_noise(ze,(rr,cc))
    s[i].id=i
for i in range(0, 4):
    Server.client_keys[i]=s[i].aggregator.public_key()
for i in range(0,4):
    s[i].aggregator.prepare_weights_D(Server.client_keys,i)
    

Server_float = server_float.secaggserver("127.0.0.1", 2019, 4, 4)
s_float = []
for i in range(0, 4):
        s_float.append(client_float.secaggclient("127.0.0.1",2019))
        s_float[i].set_noise(ze,(rr,cc))
        s_float[i].id=i    

################################################aggre##############################################################

def worker_loop(input_model, ps_weights, client_weights, client_errors,
                client_velocities, batches_queue, results_queue, gradients_queue, fedavg_lr,
                rank, world_size, compute_loss_train, compute_loss_val,
                args):
    torch.cuda.set_device(rank - 1)
    torch.random.manual_seed(args.seed)
    global numCS,numCSk,CS,CSk
    model = input_model.to(args.device)
    if args.typ==0:
        numIBLT = int(args.num_buckets)
    if args.typ==1 or args.typ==5:
        numCS = int(args.num_buckets*12.25/8)
        numCSk = int((args.num_buckets*12.25*3-args.k*4)/8/3)
    if args.typ==2 or args.typ==6:
        numCS = int(3*args.num_buckets*12.25/8)
        numCSk = int((3*args.num_buckets*12.25*3-args.k*4)/8/3)
    if args.typ==3 or args.typ==7:
        numCS = int(10*args.num_buckets*12.25/8)
        numCSk = int((10*args.num_buckets*12.25*3-args.k*4)/8/3)
    if args.typ==4 or args.typ==8:
        numCS = int(0.5*args.num_buckets*12.25/8)
        numCSk = int((0.5*args.num_buckets*12.25*3-args.k*4)/8/3)
    
    

    if args.typ==0:
        for i in range(0,4):
            IBLT_arr.append(CSVec(dd, numIBLT, rr))
    if args.typ==1 or args.typ==2 or args.typ==3 or args.typ==4:
        CS = CSVec(dd, numCS, rr)
    if args.typ==5 or args.typ==6 or args.typ==7 or args.typ==8:
        CSk = CSVec(dd, numCSk, rr)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    torch.distributed.init_process_group("nccl", rank=rank,
                                         world_size=world_size)


    while True:
        try:
            # batches is a list of batches that we should process
            # as if we were different workers for each batch
            # each batch in batches will have data belonging to a
            # single client (asserted in process_batch)
            batches = batches_queue.get(timeout=1800)
        except queue.Empty:
            print("batch queue was empty")
            return
        if batches is None:
            # reached the end of training
            break
        # get the latest weights from the parameter server
        local_ps_weights = ps_weights.clone().to(args.device)

        # sum the gradient over all batches
        if args.mode in ["uncompressed", "true_topk","nips",
                         "local_topk", "fedavg"]:
            shape = (args.grad_size,)
        elif args.mode == "sampling":
            shape = (args.grad_size*3+1,)
        elif args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        sum_g = torch.zeros(shape).to(args.device).float()

#########################################################################################################
        # print("shape",shape)
#########################################################################################################

        # first batch, first tensor (client_indices), first datum
        is_train = batches[0][0][0] != -1

        # this is the starting learning rate (which possibly decays) when
        # carrying out fedavg
        lr = fedavg_lr.to(args.device)

        all_results = []

        ###################################################################################
        grads = []
        cnt = 0
        group_size = 4
        
        ###################################################################################

        # loop over workers we have to process (see comment above)


        for batch in batches:
            print('.', end='', flush=True)
            if args.mode == "fedavg" and is_train:
                assert args.error_type == "none"
                assert args.local_momentum == 0

                original_ps_weights = local_ps_weights.clone()
                # split "batch", which is this client's entire dataset,
                # into smaller batches to run local SGD on
                if args.fedavg_batch_size == -1:
                    local_batches = [batch]
                    n_batches = 1
                else:
                    local_batches = [torch.split(t, args.fedavg_batch_size)
                                     for t in batch]
                    n_batches = len(local_batches[0])
                    local_batches = [tuple(split[i]
                                           for split in local_batches)
                                     for i in range(n_batches)]

                n_steps = n_batches * args.num_fedavg_epochs
                step = 0
                accum_results = None
                for epoch in range(args.num_fedavg_epochs):
                    for local_batch in local_batches:
                        g, results = process_batch(
                            local_batch, model, local_ps_weights,
                            client_weights,
                            client_errors, client_velocities,
                            compute_loss_train, compute_loss_val, args
                        )
                        if accum_results is None:
                            accum_results = results
                        else:
                            # accumulate results
                            for i in range(len(accum_results)):
                                accum_results[i] += results[i]
                        # g is the sum of gradients over examples, but
                        # we need to update the model with the avg grad
                        g /= local_batch[0].size()[0]
                        decay = args.fedavg_lr_decay ** step
                        local_ps_weights -= g * lr * decay
                        step += 1

                # compute average results from accum_results
                results = [r / n_steps for r in accum_results]
                g = original_ps_weights - local_ps_weights
                # weight by the batch size (which in the case of fedavg
                # is the client's dataset size) so that clients without
                # much data are downweighted
                g *= batch[0].size()[0]

                # reset local_ps_weights so that if this process has
                # to process another worker batch, the next worker
                # starts from the correct weights
                local_ps_weights[:] = original_ps_weights[:]

            else:
                # for all non-fedavg modes, we just do a single step
                if args.do_test:
                    
                    if is_train:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(
                            1.0 for _ in range(args.num_results_train))
                    else:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(
                            1.0 for _ in range(args.num_results_val))
                else:
                    g, results = process_batch(
                        batch, model, local_ps_weights, client_weights,
                        client_errors, client_velocities,
                        compute_loss_train, compute_loss_val, args,
                    )

            if is_train:
                
                grads.append(g)
                if cnt == group_size-1:
                    if args.typ==1 or args.typ==2 or args.typ==3 or args.typ==4:
                        sum_g += CS_compress(grads,args)
                    elif args.typ==5 or args.typ==6 or args.typ==7 or args.typ==8:
                        sum_g += CSk_compress(grads,args)
                    elif args.typ==0:
                        sum_g += IBLT_compress(grads,dd,group_size)
                    grads = []
                cnt = (cnt+1)%group_size
                
                
            all_results.append(results)

        
        cnt=0
        grads=[]
        print("")

        results_queue.put(all_results)



        if is_train:
            gradients_queue.put(sum_g)
            
        ################################################################################
        # if is_train:
        #     # reduce the locally summed g across devices
        #     torch.distributed.reduce(sum_g, 0)
        ################################################################################


def IBLT_compress(grads,lenth,group_size):
    ori_grad = torch.zeros(dd).to('cuda')
    cnt=0
    for grad in grads:
        IBLT_arr[cnt].zero()
        IBLT_arr[cnt].accumulateVec(grad[:dd])
        ori_grad+=grad[:dd]
        cnt+=1

    id_arr=[]
    fingerprint_arr=[]
    counter_arr=[]
    value_arr=[]
    for i in range(0,group_size):
        id_arr.append(IBLT_arr[i].id)
        fingerprint_arr.append(IBLT_arr[i].fingerprint)
        counter_arr.append(IBLT_arr[i].counter)
        value_arr.append(IBLT_arr[i].value)


    print("aggregate start",end=" ")
   
    id_aggre = aggre(id_arr,rr,cc,group_size).long()
    fingerprint_aggre = aggre(fingerprint_arr,rr,cc,group_size).long()
    counter_aggre = aggre(counter_arr,rr,cc,group_size).long()
    value_aggre = aggre_float(value_arr,rr,cc,group_size)

    print("aggregate end")
    dout = data()
    iblt = IBL_T()
    
            
    nz = ori_grad.nonzero()
    nz = nz.squeeze(1)
    nz_bincount = nz.bincount(minlength=dd).cuda().long()      

    ori_grad = ori_grad.cpu().numpy().tolist()
    ori_grad = (c_double * len(ori_grad))(*ori_grad)
    nz_bincount = nz_bincount.cpu().numpy().tolist()
    nz_bincount = (c_double * len(nz_bincount))(*nz_bincount)

        # print(IBLT.counter)
    id1=id_aggre.to('cpu').numpy()
    iblt.id1 = Convert2DToCArray(c_int,int, id1, cc)
        # id1=list(map(int,id1))
    fingerprint=fingerprint_aggre.cpu().numpy()
    iblt.fingerprint = Convert2DToCArray(c_int,int, fingerprint, cc)
    counter=counter_aggre.cpu().numpy()
    iblt.counter = Convert2DToCArray(c_int,int, counter, cc)
    value=value_aggre.cpu().numpy()
    iblt.value = Convert2DToCArray(c_double,float, value, cc)
    hash_fing=IBLT_arr[0].hash_fing.cpu().numpy().tolist()
    hash_fing=list(map(int,hash_fing))
    iblt.hash_fing = (c_int * len(hash_fing))(*hash_fing)
    buckets=IBLT_arr[0].buckets.cpu().numpy()
    iblt.buckets = Convert2DToCArray(c_int,int, buckets, dd)

        # buckets=list(map(int,buckets))
    iblt.r = IBLT_arr[0].r
    iblt.c = IBLT_arr[0].c
    iblt.d = IBLT_arr[0].d
    dll = ctypes.cdll.LoadLibrary('./c.so')
    dll.DDecode(ctypes.byref(iblt), ctypes.byref(dout),ori_grad,nz_bincount,lenth)
    # print("output_k:",dout.k)

    idx = np.frombuffer(dout.idx, dtype=np.int32)
    gradd = np.frombuffer(dout.grad, dtype=np.float64)
    idx=torch.from_numpy(idx).to('cuda')
    idx=idx.long()
    gradd=torch.from_numpy(gradd).to('cuda')
    idx=idx[:dout.k]
    gradd=gradd[:dout.k]
    # print("idx_size:",idx.size())
    
    result = torch.zeros_like(grad).to('cuda')
    result = result.double()
    # print(result.size())
    result[idx]=gradd

    return result


def CS_compress(grads,args):
    global numCS,numCSk,CS,CSk
    CS.zero()
    if args.mode in ["uncompressed", "true_topk","nips",
                         "local_topk", "fedavg"]:
        for grad in grads:
            CS.accumulateCS(grad)
        return CS._findAllValues().to('cuda')    

    if args.mode == "sampling":
        tmp = torch.zeros(dd*2+1).to('cuda')

        for grad in grads:
            CS.accumulateCS(grad[:dd])
            tmp += grad[dd:]
        return torch.cat((CS._findAllValues(),tmp)).to('cuda')

def CSk_compress(grads,args):
    global numCS,numCSk,CS,CSk
    CSk.zero()
    if args.mode in ["uncompressed", "true_topk","nips",
                         "local_topk", "fedavg"]:
        ori_grad = torch.zeros(dd).to('cuda')
        
        for grad in grads:
            CSk.accumulateCS(grad)
            ori_grad += grad

        x = CSk._findAllValues()
        nz = ori_grad.nonzero()
        nz = nz.squeeze(1)
        nz_bincount = nz.bincount(minlength=dd).cuda().long()
        x = x*nz_bincount
        return x.to('cuda')

    if args.mode == "sampling":
        ori_grad = torch.zeros(dd).to('cuda')
        tmp = torch.zeros(dd*2+1).to('cuda')
        for grad in grads:
            CSk.accumulateCS(grad[:dd])
            ori_grad += grad[:dd]
            tmp += grad[dd:]

        x = CSk._findAllValues()
        nz = ori_grad.nonzero()
        nz = nz.squeeze(1)
        nz_bincount = nz.bincount(minlength=dd).cuda().long()
        x = x*nz_bincount
        return torch.cat((x,tmp)).to('cuda')



def Convert1DToCArray(TYPE,type1, ary):
    tmp = ary.tolist()
    tmp = list(map(type1,tmp))
    arow = TYPE(*tmp)
    return arow
def Convert2DToCArray(TYPE,type1, ary, llen):
    # ROW = c_double * len(ary[0])
    ROW = TYPE * llen
    # ROW = TYPE * len(ary[0])
    rows = []
    for i in range(len(ary)):
        rows.append(Convert1DToCArray(ROW,type1, ary[i]))
    MATRIX = ROW * len(ary)
    return MATRIX(*rows)



def aggre_float(id_arr,rr,cc,group_size):
    # f=open("aggre.txt","a")

    for i in range(0,group_size):
        s_float[i].weight = s_float[i].aggregator.add_noise(id_arr[i],i)

    Server_float.aggregate=torch.zeros(rr,cc).to('cuda')
    for i in range(0,group_size):
        Server_float.aggregate += s_float[i].weight

    # f.close()
    return Server_float.aggregate

def aggre(id_arr,rr,cc,group_size):
    # f=open("aggre.txt","a")

    for i in range(0,group_size):       
        s[i].weight = s[i].aggregator.add_noise(id_arr[i],i)

    Server.aggregate=torch.zeros(rr,cc).to('cuda').long()
    for i in range(0,group_size):
        Server.aggregate += s[i].weight


    # f.close()
    return Server.aggregate



def process_batch(batch, model, ps_weights, client_weights,
                  client_errors, client_velocities,
                  compute_loss_train, compute_loss_val, args):
    client_indices = batch[0]
    is_train = client_indices[0] != -1
    batch = batch[1:]
    batch = [t.to(args.device) for t in batch]
    assert (client_indices - client_indices[0]).abs().sum() == 0
    client_id = client_indices[0]

    # figure out what model weights this worker should use
    new_worker_weights = None
    if args.do_topk_down:
        worker_weights = client_weights[client_id].to(args.device)
        new_worker_weights = get_new_worker_weights(ps_weights,
                                                    worker_weights,
                                                    args)
        new_worker_weights = new_worker_weights.to(args.device)
    else:
        new_worker_weights = ps_weights

    # get model ready
    set_param_vec(model, new_worker_weights)

    transmit = None
    if is_train:
        model.train()
        model.zero_grad()
        # get our client's local velocity & local error vectors
        velocity = None
        error = None
        if client_velocities is not None:
            velocity = client_velocities[client_id].to(args.device)
        if client_errors is not None:
            error = client_errors[client_id].to(args.device)

        results, transmit = local_step(model, velocity, error, batch,
                                       compute_loss_train, args)
        ################################################################################
        # print(transmit.shape, (transmit != torch.zeros(transmit.shape).cuda()).sum())
        ################################################################################
    else:
        model.eval()
        results = forward_grad(model, batch, compute_loss_val, args,
                               compute_grad=False)

    return transmit, results


def local_step(model, velocity, error, batch, compute_loss, args):
    # g is a (possibly compressed) gradient
    g, results = forward_grad(model, batch, compute_loss, args)

    # locally, we need to deal with the sum of gradients across
    # examples, since we will torch.distributed.reduce the to_transmits,
    g *= batch[0].size(0)

    # if needed, do local momentum
    if args.local_momentum > 0:
        # this does velocity[:] = m * velocity + g, but twice as fast
        torch.add(g, velocity, alpha=args.local_momentum, out=velocity)

    # if needed, do local error correction
    if args.error_type == "local":
        error += velocity if velocity is not None else g
        to_transmit = error
    else:
        to_transmit = velocity if velocity is not None else g

    if args.mode == "local_topk":
        assert args.error_type in ["local", "none"]
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k)

        nz = to_transmit.nonzero()
        if error is not None:
            # error feedback
            error[nz] = 0

        # if we're doing local momentum, do momentum factor masking
        if args.local_momentum > 0:
            velocity[nz] = 0

    if args.mode == "sampling":
        assert args.error_type in ["local", "none"]
        to_transmit = _sampling(to_transmit.to(args.device), k=args.k)

        nz = to_transmit.nonzero()
        if error is not None:
            error[nz] = 0

        if args.local_momentum > 0:
            velocity[nz] = 0

    if args.mode == "nips":
        assert args.error_type in ["local", "none"]
        to_transmit = _nips(to_transmit.to(args.device), k=args.k)

        nz = to_transmit.nonzero()
        if error is not None:
            error[nz] = 0

        if args.local_momentum > 0:
            velocity[nz] = 0
    # sketched sgd with local error accumulation doesn't really make
    # sense, since when we send a sketch we don't know what portion
    # of the sketch is the "error"
    if error is not None:
        assert args.mode not in ["sketch", "uncompressed"]

    # we want to do momentum factor masking for all the compression
    # methods, but that's not possible to do for sketching, since
    # it's unknown which coordinates to mask out
    if velocity is not None:
        assert args.mode != "sketch"

    return results, to_transmit


def get_new_worker_weights(ps_weights, worker_weights, args):
    device = args.device

    ps_weights = ps_weights.to(device)
    worker_weights = worker_weights.to(device)

    # we'll update the old worker_weights with a possibly compressed
    # version of diff_vec
    diff_vec = ps_weights - worker_weights
    if args.do_topk_down:
        weight_update = _topk(diff_vec, k=args.k)
    else:
        weight_update = diff_vec

    new_worker_weights = worker_weights + weight_update
    return new_worker_weights


def forward_grad(model, batch, compute_loss, args, compute_grad=True):
    device = args.device

    # divide up batch (for gradient accumulation when memory constrained)
    # num_shards = args.num_train_batch_shards
    # need the max(1, ...) since the last batch in an epoch might be small
    # microbatch_size = max(1, batch[0].size()[0] // num_shards)
    if args.microbatch_size > 0:
        microbatch_size = min(batch[0].size()[0], args.microbatch_size)
    else:
        microbatch_size = batch[0].size()[0]

    # accumulators for the loss & metric values
    accum_loss = 0
    accum_metrics = None

    num_iters = math.ceil(batch[0].size()[0] / microbatch_size)
    for i in range(num_iters):
        # extract current microbatch
        start = i * microbatch_size
        end = (i+1) * microbatch_size
        microbatch = [t[start:end] for t in batch]

        # forward pass
        loss, *metrics = compute_loss(model, microbatch, args)

        # if first time through, we find out how many metrics there are
        if accum_metrics is None:
            accum_metrics = [0 for _ in metrics]

        # accumulate loss & metrics, weighted by how many data points
        # were actually used
        accum_loss += loss.item() * microbatch[0].size()[0]
        for i, m in enumerate(metrics):
            accum_metrics[i] += m.item() * microbatch[0].size()[0]

        # backward pass
        if compute_grad:
            loss.backward()

    # gradient clipping
    if compute_grad and args.max_grad_norm is not None and args.mode not in ["sketch"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       args.max_grad_norm * num_iters)

    # "average" here is over the data in the batch
    average_loss = accum_loss / batch[0].size()[0]
    average_metrics = [m / batch[0].size()[0] for m in accum_metrics]

    results = [average_loss] + average_metrics

    if not compute_grad:
        return results

    grad = get_grad(model, args)
    if args.do_dp:
        grad = clip_grad(args.l2_norm_clip, grad)
        if args.dp_mode == "worker":
            noise = torch.normal(
                mean=0, std=args.noise_multiplier, size=grad.size()).to(args.device)
            noise *= np.sqrt(args.num_workers)
            grad += noise

    # compress the gradient if needed
    if args.mode == "sketch":
        sketch = CSVec(d=args.grad_size, c=args.num_cols,
                       r=args.num_rows, device=args.device,
                       numBlocks=args.num_blocks)
        sketch.accumulateVec(grad)
        # gradient clipping
        if compute_grad and args.max_grad_norm is not None:
            sketch = clip_grad(args.max_grad_norm, sketch)
        g = sketch.table
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        # ideally we'd return the compressed version of the gradient,
        # i.e. _topk(grad, k=args.k). However, for sketching we do momentum
        # in the sketch, whereas for topk we do momentum before taking topk
        # so we have to return an inconsistent quantity here
        g = grad
    elif args.mode == "sampling":
        g = grad
    elif args.mode == "fedavg":
        # logic for doing fedavg happens in process_batch
        g = grad
    elif args.mode == "uncompressed":
        g = grad
    elif args.mode == "nips":
        g = grad
    return g, results
