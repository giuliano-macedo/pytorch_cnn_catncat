#!/usr/bin/env python3
from args import get_args
import os
from utils import download_and_extract_zip,get_dataset
import torch
import torch.nn as nn
from graph_animation import Animation

torch.random.manual_seed(1)
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("runs")


args=get_args()

if args.gpu and (not torch.cuda.is_available()):
	raise RuntimeError("Cuda is not supported")

if args.dataset==None:
	args.dataset="dataset"
	if not os.path.isdir("dataset"):
		url="https://llpinokio-ia.herokuapp.com/cats.zip"
		print("downloading dataset...")
		download_and_extract_zip(url,args.dataset)

train_imgs,train_labels=get_dataset(args.dataset,"train",args.gpu,shuffle=True)
eval_imgs,eval_labels=get_dataset(args.dataset,"evaluation",args.gpu)

# print(train_imgs[0])
# print(train_imgs.size())
# exit()

model=args.model
if args.gpu:
	model.cuda()

no_train=float(train_labels.size(0))
no_eval=float(eval_labels.size(0))

if args.gpu:
	model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

anim_accr=Animation(args.no_epochs,title="Accuracy")
anim_loss=Animation(args.no_epochs,title="Loss")

for epoch in range(args.no_epochs):
	optimizer.zero_grad()
	
	model.train()
	outputs = model(train_imgs)
	loss_train = criterion(outputs, train_labels)
	_, predicted = torch.max(outputs.data, 1)
	accr_train=((predicted == train_labels).sum())/no_train

	loss_train.backward()
	optimizer.step()

	model.eval()
	optimizer.zero_grad()
	outputs = model(eval_imgs)
	loss_eval = criterion(outputs, eval_labels)
	_, predicted = torch.max(outputs.data, 1)
	accr_eval=((predicted == eval_labels).sum())/no_eval

	print(f"{epoch} train accr:{accr_train*100:.2f}%  eval accr:{accr_eval*100:.2f}% train loss:{loss_train:e} eval loss:{loss_eval:e}")

	anim_loss.update(float(loss_train),float(loss_eval))
	anim_accr.update(float(accr_train),float(accr_eval))

# 	writer.add_scalar('Loss/train', loss_train, epoch)
# 	writer.add_scalar('Loss/test', loss_eval, epoch)
# 	writer.add_scalar('Accuracy/train', accr_train, epoch)
# 	writer.add_scalar('Accuracy/test', accr_eval, epoch)
# writer.close()