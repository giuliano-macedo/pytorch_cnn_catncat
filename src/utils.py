from urllib.request import urlopen
import zipfile
import io
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch
import os
from glob import glob
import random
random.seed(1)

def download_file(url):
	def download(res):
		while True:
			chunk = res.read(1000)
			if not chunk:
				return
			yield chunk
	ans=bytearray()
	res = urlopen(url)
	total=int(res.info().get('content-length', 0))
	t=tqdm(total=total,unit='B', unit_scale=True)
	for chunk in download(res):
		t.update(len(chunk))
		ans+=chunk
	t.close()
	return ans
def download_and_extract_zip(url,path):
	f=io.BytesIO(download_file(url))
	my_zip=zipfile.ZipFile(f)
	my_zip.extractall(path)
def get_dataset(root,dataset_type,shuffle=False):
	imgs=[]
	labels=[]

	for y,y_label in enumerate(["noncat","cat"]):
		files=glob((os.path.join(root,dataset_type,y_label,"*.png")))
		if shuffle:random.shuffle(files)
		for fname in files:
			img=np.expand_dims(np.array(Image.open(fname),dtype=np.float32).mean(axis=2)/255.0,axis=0)
			imgs.append(img)
			labels.append(y)

	return Variable(torch.tensor(imgs)),Variable(torch.tensor(labels))
	