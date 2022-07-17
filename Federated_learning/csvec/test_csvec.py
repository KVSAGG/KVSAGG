import unittest
import csvec
from csvec import CSVec
import torch
import ctypes
from ctypes import *
import sys
sys.path.append("..")
from secure_aggre import client
from secure_aggre import server
import numpy as np
class Base:
    # use Base class to hide CSVecTestCase from the unittest runner
    # we only want the subclasses to actually be run

    class CSVecTestCase(unittest.TestCase):
        def testRandomness(self):
            # make sure two sketches get the same hashes and signs
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            b = CSVec(d, c, r, **self.csvecArgs)
            self.assertTrue(torch.allclose(a.signs, b.signs))
            self.assertTrue(torch.allclose(a.buckets, b.buckets))
            self.assertTrue(torch.allclose(a.signs, b.signs))

            if self.numBlocks > 1:
                self.assertTrue(torch.allclose(a.blockOffsets,
                                               b.blockOffsets))
                self.assertTrue(torch.allclose(a.blockSigns,
                                               b.blockSigns))

        def testInit(self):
            # make sure the table starts out zeroed
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            zeros = torch.zeros(r, c).to(self.device)
            self.assertTrue(torch.allclose(a.table, zeros))

        def testSketchVec(self):
            # sketch a vector with all zeros except a single 1
            # then the table should be zeros everywhere except a single
            # 1 in each row
            d = 100
            c = 1
            r = 5
            a = CSVec(d=d, c=c, r=r, **self.csvecArgs)
            vec = torch.zeros(d).to(self.device)
            vec[0] = 1
            a.accumulateVec(vec)
            # make sure the sketch only has one nonzero entry per row
            for i in range(r):
                with self.subTest(row=i):
                    self.assertEqual(a.table[i,:].nonzero().numel(), 1)

            # make sure each row sums to +-1
            summed = a.table.abs().sum(dim=1).view(-1)
            ones = torch.ones(r).to(self.device)
            self.assertTrue(torch.allclose(summed, ones))

        def testZeroSketch(self):
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            vec = torch.rand(d).to(self.device)
            a.accumulateVec(vec)

            zeros = torch.zeros((r, c)).to(self.device)
            self.assertFalse(torch.allclose(a.table, zeros))

            a.zero()
            self.assertTrue(torch.allclose(a.table, zeros))

        def testUnsketch(self):
            # make sure heavy hitter recovery works correctly

            # use a gigantic sketch so there's no chance of collision
            d = 5
            c = 10000
            r = 20
            a = CSVec(d, c, r, **self.csvecArgs)
            vec = torch.rand(d).to(self.device)

            a.accumulateVec(vec)

            with self.subTest(method="topk"):
                recovered = a.unSketch(k=d)
                self.assertTrue(torch.allclose(recovered, vec))

            with self.subTest(method="epsilon"):
                thr = vec.abs().min() * 0.9
                recovered = a.unSketch(epsilon=thr / vec.norm())
                self.assertTrue(torch.allclose(recovered, vec))

        def testSketchSum(self):
            d = 5
            c = 10000
            r = 20

            summed = CSVec(d, c, r, **self.csvecArgs)
            for i in range(d):
                vec = torch.zeros(d).to(self.device)
                vec[i] = 1
                sketch = CSVec(d, c, r, **self.csvecArgs)
                sketch.accumulateVec(vec)
                summed += sketch

            recovered = summed.unSketch(k=d)
            trueSum = torch.ones(d).to(self.device)
            self.assertTrue(torch.allclose(recovered, trueSum))

        def testL2(self):
            d = 5
            c = 10000
            r = 20

            vec = torch.randn(d).to(self.device)
            a = CSVec(d, c, r, **self.csvecArgs)
            a.accumulateVec(vec)

            tol = 0.0001
            self.assertTrue((a.l2estimate() - vec.norm()).abs() < tol)

        def testMedian(self):
            d = 5
            c = 10000
            r = 20

            csvecs = [CSVec(d, c, r, **self.csvecArgs) for _ in range(3)]
            for i, csvec in enumerate(csvecs):
                vec = torch.arange(d).float().to(self.device) + i
                csvec.accumulateVec(vec)
            median = CSVec.median(csvecs)
            recovered = median.unSketch(k=d)
            trueMedian = torch.arange(d).float().to(self.device) + 1
            self.assertTrue(torch.allclose(recovered, trueMedian))

class TestCaseCPU1(Base.CSVecTestCase):
    def setUp(self):
        # hack to reset csvec's global cache between tests
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 1

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

class TestCaseCPU2(Base.CSVecTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestCaseCUDA2(Base.CSVecTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cuda"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

dd = 4000
cc = 30000 # 2^19
rr = 3
IBLT = CSVec(dd, cc, rr)
IBLT_arr = []
for i in range(0,4):
    IBLT_arr.append(CSVec(dd, cc, rr))
print("IBLT initialization finished")
class IBL_T(ctypes.Structure):
    _fields_ = [("r",c_int),("c",c_int),("d",c_int),("id1",(c_int*cc)*rr),("fingerprint",(c_int*cc)*rr),("counter",(c_int*cc)*rr),("value",(c_double*cc)*rr),
    ("hash_fing",c_int*dd),("buckets",(c_int*dd)*rr)]
class data(ctypes.Structure):
    _fields_ = [("k",c_int), ("idx",c_int * 3000010), ("grad", c_double * 3000010)]

def aggre(id_arr,rr,cc,group_size):
    Server = server.secaggserver("127.0.0.1", 2019, 3, 2)
    s = []
    for i in range(0,group_size):
        s.append(client.secaggclient("127.0.0.1",2019))
    for i in range(0,group_size):
        s[i].set_weights(id_arr[i].unsqueeze(0).double(),(rr,cc))
        s[i].id=i
        print(i,"weights",s[i].aggregator.weights)

    for i in range(0,group_size):
        Server.client_keys[i]=s[i].aggregator.public_key()
        print(i,Server.client_keys[i])

    for i in range(0,group_size):
        s[i].weight = s[i].aggregator.prepare_weights(Server.client_keys,i)
        s[i].weight = s[i].weights_encoding(s[i].weight)

    for i in range(0,group_size):
        print(i,"final weight",Server.weights_decoding(s[i].weight))
        Server.aggregate += Server.weights_decoding(s[i].weight).squeeze(0)

    # for i in range(0,group_size):
    #     Server.aggregate += Server.weights_decoding( s[i].weights_encoding(-1*s[i].aggregator.private_secret()) )

    return Server.aggregate


# vec = torch.tensor([[1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1],
#                 [1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,0],
#                 [0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,0,1,0,1,0],
#                 [0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1]]).to('cuda')
# vec1 = torch.tensor([[1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1],
#                 [1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1],
#                 [0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0],
#                 [0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1]]).to('cuda')
# vec2 = torch.tensor([[1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0],
#                 [1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,],
#                 [0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0],
#                 [0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,1,1]]).to('cuda')
vec = torch.tensor( np.int32(np.random.rand(4,dd)) %20).to('cuda')
print(vec)
cnt=0
for v in vec:
    # print(v)
    IBLT.accumulateVec(v)
    cnt+=1

# cnt=0
# for v in vec1:
#     # print(v)
#     IBLT.accumulateVec(v)
#     cnt+=1
# cnt=0
# for v in vec2:
#     # print(v)
#     IBLT.accumulateVec(v)
#     cnt+=1
# print("IBLT id",IBLT.id)
# print("IBLT finger",IBLT.fingerprint)
# print("IBLT counter",IBLT.counter)
# print("IBLT value",IBLT.value)


# id_arr=[]
# for i in range(0,4):
#     id_arr.append(IBLT_arr[i].id)
# id_aggre = aggre(id_arr,rr,cc,4)
# print("id_aggre",id_aggre)
# id_sum=torch.zeros(rr,cc).to('cuda')
# for k in range(0,4):
#     # id_sum+=id_arr[k]
#     id_sum+=IBLT_arr[k].id
# print("id_sum",id_sum)
# id_aggre -= id_sum
# print(id_aggre.nonzero())

def Convert1DToCArray(TYPE,type1, ary):
    tmp = ary.tolist()
    tmp = list(map(type1,tmp))
    arow = TYPE(*tmp)
    return arow
def Convert2DToCArray(TYPE,type1, ary):
    # ROW = c_double * len(ary[0])
    ROW = TYPE * len(ary[0])
    rows = []
    for i in range(len(ary)):
        rows.append(Convert1DToCArray(ROW,type1, ary[i]))
    MATRIX = ROW * len(ary)
    return MATRIX(*rows)



# def IBLT_and_aggre_GPU(grads, k, group_size):
#     # print("Inserting")
#     # IBLT_arr = []
#     # for i in range(0,group_size):
#     #     IBLT_arr.append(CSVec(dd, cc, rr))
#     cnt=0
#     # IBLT_arr[0].accumulateVec(grads[0][:6568640])
#     # print(IBLT_arr[0])
#     # print(IBLT)
#     for grad in grads:
#         IBLT_arr[cnt].accumulateVec(grad[:6568640])
#         cnt+=1

#     id_arr=[]
#     for i in range(0,group_size):
#         id_arr.append(IBLT_arr[i].id)
#     id_aggre = aggre(id_arr,rr,cc,group_size)
#     # print("id_aggre",id_aggre)
#     id_sum=torch.zeros(rr,cc).to('cuda')
#     for k in range(0,group_size):
#         id_sum+=IBLT_arr[i].id
#     # print("id_sum",id_sum)
#     id_aggre -= id_sum
#     print(torch.nonzero(id_aggre))

#     IBLT.zero()
#######################################################################
dout = data()
iblt = IBL_T()    
id1=IBLT.id.cpu().numpy()
iblt.id1 = Convert2DToCArray(c_int,int, id1)
    # id1=list(map(int,id1))
fingerprint=IBLT.fingerprint.cpu().numpy()
iblt.fingerprint = Convert2DToCArray(c_int,int, fingerprint)
counter=IBLT.counter.cpu().numpy()
iblt.counter = Convert2DToCArray(c_int,int, counter)
value=IBLT.value.cpu().numpy()
iblt.value = Convert2DToCArray(c_double,float, value)
hash_fing=IBLT.hash_fing.cpu().numpy().tolist()
hash_fing=list(map(int,hash_fing))
iblt.hash_fing = (c_int * len(hash_fing))(*hash_fing)
buckets=IBLT.buckets.cpu().numpy()
iblt.buckets = Convert2DToCArray(c_int,int, buckets)
    # buckets=list(map(int,buckets))
iblt.r = IBLT.r
iblt.c = IBLT.c
iblt.d = IBLT.d

# for i in range(0,rr):
#     for j in range(0,cc):
#         print(iblt.counter[i][j],end=' ')
#     print('\n')


# dll = ctypes.cdll.LoadLibrary('../c.so')
# dll.DDecode(ctypes.byref(iblt), ctypes.byref(dout))
# print("output_k:",dout.k)
# IBLT.zero()

#######################################################################
