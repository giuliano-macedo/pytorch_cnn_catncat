import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process,Queue
class Activity(Process):
	def __init__(self,n,queue,title=""):
		self.n=n
		self.queue=queue
		self.title=title
		super(Activity, self).__init__()
	def run(self):
		self.train_x, self.train_y = [], []
		self.eval_x, self.eval_y = [], []
		self.fig, self.ax = plt.subplots()
		self.ln_train, = plt.plot([], [], label="train")
		self.ln_eval,  = plt.plot([], [],  label="evaluation")
		FuncAnimation(self.fig, self.__anim_update,interval=16, init_func=self.__anim_init, blit=True)
		plt.title(self.title)
		plt.show()
	def __anim_init(self):
		self.ax.set_xlim(0, self.n)
		self.ax.legend()
		return self.ln_train,self.ln_eval
	
	def __anim_update(self,frame):
		train=self.queue.get()
		self.train_y.append(train)
		self.train_x.append(len(self.train_y))
		self.ln_train.set_data(self.train_x, self.train_y)
		
		_eval=self.queue.get()
		self.eval_y.append(_eval)
		self.eval_x.append(len(self.eval_y))
		self.ln_eval.set_data(self.eval_x, self.eval_y)
		
		# self.ax.set_ylim(min(self.eval_y+self.train_y), max(self.eval_y+self.train_y))
	
		return self.ln_train,self.ln_eval

class Animation():
	def __init__(self,n,title=""):
		self.n=n
		self.queue=Queue()
		self.actvity=Activity(self.n,self.queue,title)
		self.actvity.start()
	def update(self,train,_eval):
		# print(f"sending {train}")
		self.queue.put(train)
		self.queue.put(_eval)
	def __del__(self):
		self.actvity.join()


if __name__=="__main__":
	import time
	anim1=Animation(60*10,title="anim1")
	anim2=Animation(60*10,title="anim2")
	for i in range(60*10):
		anim1.update((np.sin(i/10)+1)*.5,(np.cos(i/10)+1)*.5)
		anim2.update((np.sin(i/10)+1)*.5,(np.cos(i/10)+1)*.5)
	# 	time.sleep(.017)
	