# from socketIO_client import SocketIO, LoggingNamespace
from random import randrange
import random
import datetime
import numpy as np
from copy import deepcopy
import codecs
import pickle
import json
import torch


class SecAggregator:
	def __init__(self,common_base,common_mod,dimensions,weights):
        # random.seed(dateime.datetime.now())
		# self.secretkey = randrange(common_mod)
		self.secretkey = random.randint(1,common_mod)
		self.base = common_base
		self.mod = common_mod
		self.pubkey = (self.base**self.secretkey) % self.mod
		self.sndkey = randrange(common_mod)
		self.dim = dimensions
		self.weights = weights
		self.keys = {}
		self.id = ''
		self.noise = 0
	def public_key(self):
		return self.pubkey
	def set_noise(self,noise,dims):
		self.noise = noise.to('cuda').long()
		self.dim = dims
	def set_weights(self,wghts,dims):
		self.weights = wghts.to('cuda').long()
		self.dim = dims
	def configure(self,base,mod):
		self.base = base
		self.mod = mod
		self.pubkey = (self.base**self.secretkey) % self.mod
	def generate_weights(self,seed):
		np.random.seed(seed)
		return torch.tensor( np.random.randint(0,100,size=(self.dim[0],self.dim[1]))).to('cuda').long()
		# return torch.tensor(np.int32( np.random.randint(0,100,size=(self.dim[0],self.dim[1])))).to('cuda')
		# return torch.tensor( np.float32(np.random.rand(self.dim[0],self.dim[1])) ).to('cuda')


	def add_noise(self,wghts,myid):
		return wghts.to('cuda').long() + self.noise

	def prepare_weights_D(self,shared_keys,myid):
		self.keys = shared_keys
		self.id = myid
		# print("type",wghts.type())
		# f=open("aggre.txt","a")
		for sid in shared_keys:
			if sid>myid:
				# print ("1",myid,sid,self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod),file=f)
				self.noise+=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
			elif sid<myid:
				# print ("2",myid,sid,self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod),file=f)
				self.noise-=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
		# wghts+=self.generate_weights(self.sndkey)
		# f.close()
		

	def prepare_weights(self,shared_keys,myid):
		self.keys = shared_keys
		self.id = myid
		wghts = deepcopy(self.weights)
		# print("type",wghts.type())
		# f=open("aggre.txt","a")
		for sid in shared_keys:
			if sid>myid:
				# print ("1",myid,sid,self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod),file=f)
				wghts+=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
			elif sid<myid:
				# print ("2",myid,sid,self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod),file=f)
				wghts-=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
		# wghts+=self.generate_weights(self.sndkey)
		# f.close()
		return wghts
	def reveal(self, keylist):
		wghts = np.zeros(self.dim)
		for each in keylist:
			print (each)
			if each<self.id:
				wghts-=self.generate_weights((self.keys[each]**self.secretkey)%self.mod)
			elif each>self.id:
				wghts+=self.generate_weights((self.keys[each]**self.secretkey)%self.mod)
		return -1*wghts
	def private_secret(self):
		return self.generate_weights(self.sndkey)


class secaggclient:
	def __init__(self,serverhost,serverport):
		# self.sio = SocketIO(serverhost,serverport,LoggingNamespace)
		self.aggregator = SecAggregator(3,100103,(10,10),np.float32(np.full((10,10),3,dtype=int)))
		self.id = ''
		self.keys = {}

	def start(self):
		self.register_handles()
		print("Starting")
		# self.sio.emit("wakeup")
		# self.sio.wait()

	def configure(self,b,m):
		self.aggregator.configure(b,m)

	def set_weights(self,wghts,dims):
		self.aggregator.set_weights(wghts,dims)
	def set_noise(self,noise,dims):
		self.aggregator.set_noise(noise,dims)
	def weights_encoding(self, x):
		return codecs.encode(pickle.dumps(x), 'base64').decode()

	def weights_decoding(self, s):
		return pickle.loads(codecs.decode(s.encode(),'base64'))

	def register_handles(self):
		def on_connect(*args):
			msg = args[0]
			# self.sio.emit("connect")
			print("Connected and recieved this message",msg['message'])

		def on_send_pubkey(*args):
			msg = args[0]
			self.id = msg['id']
			pubkey = {
				'key': self.aggregator.public_key()
			}
			# self.sio.emit('public_key',pubkey)

		def on_sharedkeys(*args):
			keydict = json.loads(args[0])
			self.keys = keydict
			print("KEYS RECIEVED: ",self.keys)
			weight = self.aggregator.prepare_weights(self.keys,self.id)
			weight = self.weights_encoding(weight)
			resp = {
				'weights':weight
			}
			# self.sio.emit('weights',resp)

		def on_send_secret(*args):
			secret = self.weights_encoding(-1*self.aggregator.private_secret())
			resp = {
				'secret':secret
			}
			# self.sio.emit('secret',resp)			

		def on_reveal_secret(*args):
			keylist = json.loads(args[0])
			resp = {
				'rvl_secret':self.weights_encoding(self.aggregator.reveal(keylist))
			}
			# self.sio.emit('rvl_secret',resp)


		def on_disconnect(*args):
			# self.sio.emit("disconnect")
			print("Disconnected")


if __name__=="__main__":
	s = secaggclient("127.0.0.1",2019)
	s.set_weights(np.zeros((10,10)),(10,10))
	s.configure(2,100255)
	s.start()
	print("Ready")