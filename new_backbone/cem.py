import torch.nn as nn
import torch.nn.functional as F
import torch

def CEM(featurMap):
  F3,F4,F5=featurMap[-3:]
  f33x=nn.Conv2d(512,256,kernel_size=3, padding=1, stride=1).cuda()
  f33=f33x(F3) #f33
  f43x=nn.Conv2d(1024,256,kernel_size=3, padding=1, stride=1).cuda()
  f43=f43x(F4)
  f43up=F.interpolate(f43,100, mode='bilinear', align_corners=False) #f43
  f53x=nn.Conv2d(2048,256,kernel_size=3, padding=1, stride=1).cuda()
  f53=f53x(F5)
  f53up=F.interpolate(f53,100, mode='bilinear', align_corners=False) #f53

  w33x=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1).cuda()
  w33=w33x(f33)
  w43x=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1).cuda()
  w53x=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1).cuda()
  w43=w33x(f43up)
  w53=w33x(f53up)

  cat=torch.cat((w33,w43),1)
  cat=torch.cat((cat,w53),1)
  cat2x=nn.Conv2d(768,256,kernel_size=3,stride=1,padding=1).cuda()
  cat2=cat2x(cat)
  w=F.softmax(cat2,0)
  
  fm=torch.matmul(input=w,other=w53)+torch.matmul(input=w,other=w43)+torch.matmul(input=w,other=w33) #fm

  #context enhancment module CEM

  cba11=nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba13=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,3), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba31=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,1), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba15=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,5), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba51=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(5,1), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba17=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,7), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba71=nn.Sequential(nn.Conv2d(64, 64, kernel_size=(7,1), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()
  cba33=nn.Sequential(nn.Conv2d(256, 64, kernel_size=(3,3), padding='same', stride=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True)).cuda()

  fc1=cba11(fm)
  fc2=cba31(cba13(cba11(fm)))
  fc3=cba51(cba15(cba11(fm)))
  fc4=cba71(cba17(cba11(fm)))
  fc5=cba33(fm)
  fcem=torch.cat((fc1,fc2),1)
  fcem=torch.cat((fcem,fc3),1)
  fcem=torch.cat((fcem,fc4),1)
  fcem=torch.cat((fcem,fc5),1)
  return fcem
