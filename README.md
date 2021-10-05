# Resnet 을 사용한 이미지 분류 학습


딥러닝에서 CNN은 이미지 인식 분야에서 뛰어난 성능을 보여주고 있습니다. CNN 네트워크의 층(layer)을 더 쌓으며 아주 깊은 네트워크를 구현하여 성능 향상을 이룰 수 있습니다. 
하지만 실제로 어느정도 이상 깊어진 네트워크는 오히려 vanishing/exploding gradient 문제 때문에 성능이 더 떨어졌는데 이러한 현상을 degradation problem이라고 합니다. 
문제가 발생하는 이유는 weight들의 분포가 균등하지 않고, 역전파가 제대로 이루어지지 않기 때문입니다. ResNet은 이러한 neural network의 구조가 깊어질수록 정확도가 감소하는 문제를 해결하기 위해 제안되었습니다.
ResNet은 2015 년 Microsoft Research Asia의 He Kaiming, Sun Jian 등이 제안한 네트워크 구조로 ILSVRC-2015 분류 과제 에서 1 위를 차지했습니다. 동시에 ImageNet 감지, ImageNet 현지화, COCO 감지 및 COCO 분할 작업에서 1 위를 차지했습니다.
ResNet의 original 논문명은 "Deep Residual Learning for Image Recognition"이고, 저자는 Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jiao Sun으로 전부 다 중국인입니다. 
2014년의 GoogLeNet이 22개 층으로 구성된 것에 비해, ResNet은 152개 층을 갖는다. 약 7배나 깊어졌습니다.  ResNet은 연결 건너 뛰기 또는 잔여 블록을 사용하여 특정 계층을 건너 뛰어 더 깊은 네트워크에서 최적화 / 저하 문제를 해결하여 깊이를 높이는 데 성공했으며 대신 모델의 성능에 해를 끼치 지 않고 성능이 향상되었습니다.

이제 RESNET 구조를 사용하여 전이학습으로 2022년 대선 후보들의 얼굴을 인식/분류하는 학습을 테스트해 보겠습니다.


##  대선후보 얼굴 사진 크롤링

일단 대선후보들의  멤버들의 사진을 모아야 합니다. 웹크롤러는 깃허브 오픈소스 프로젝트인 AutoCrawler를 사용했습니다.
( https://github.com/YoongiKim/AutoCrawler ) 참고하세요
( https://www.opensourceagenda.com/projects/autocrawler )

먼저 
1. 다음 링크 파이썬 소스를 다운로드한다 https://github.com/YoongiKim/AutoCrawler 
2. keywords.txt에 내가 크롤링하고 싶은 검색어를 한 줄에 한 개씩 쓴다.
3. 터미널 창에서 python main.py 를 실행한다.
4. 자동으로 크롬이 실행되며 selenium 으로 이미지를 폴더별로 다운로드 합니다.
5. 프로그램이 종료 할때 까지 기다리시면 됩니다.

![Keyword](./img/keyword.png)
![download](./img/crawresult.png)


1000개 이상의 많은 이미지가 다운로드 되었습니다. 
학습의 정확도를 위해 여러 명이 나온 사진들과 같은 부적합한 이미지들을 삭제해 줍니다.
대선 후보 사진들을 크롤링해서 모으는 것보다 부적합한 사진들을 지우는데 더 오랜 시간이 걸렸습니다. 역시 양질의 데이터가 중요합니다.


## RESNET 을 이용한 이미지 분류 전이학습 및 테스트

이제는 pyTorch에서 RESNET을 사용해서 전이학습하는 소스를 만들어 봅니다.
아래 쏘스는 다음 싸이트를 참고 했습니다.   ( https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-pytorch )


* 먼저 여러가지 라이브러리를 import 합니다.
특히 torchvision의 models 를 import 하여 resnet50 모델을 사용할수 있도록 해야 합니다.

~~~
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
~~~



테스트 이미지를 학습된 모델로 분류 해봅니다.
아주 잘 작동합니다.
![result](./img/result.png)