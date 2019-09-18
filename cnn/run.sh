export PROJECT_HOME="/home/yidong.hu/study/CNN_models/cnn"
#  使用vgg16网络训练caltech256数据集
./classify.py -t first-noen -save 50 -wn ?.pkl -d caltech256 -en -shm -train -test -epochs 100 -lr 0.0001 -b 800 -lw 8 -s 224 -g 0,1,2,3,4,5,6,7

#  使用googLenet网络训练caltech256数据集
./classify.py -t second -m googLenet -save 50 -wn ?.pkl -d caltech256 -en -shm -train -test -epochs 100 -lr 0.0001 -b 800 -lw 8 -s 224 -g 0,1,2,3,4,5,6,7
./classify.py -t second -m googLenet -save 9999 -wn second_googLenet_on_caltech256_bs_800.pkl -d caltech256 -en -shm -train -test -epochs 50 -lr 0.0001 -b 1600 -lw 8 -s 224 -g 0,1,2,3,4,5,6,7


# 使用resnet50网络训练cifar10数据集
./classify.py -t second -m resnet50 -save 50 -wn ?.pkl -d cifar10 -en -train -test -epochs 10 -lr 0.001 -b 800 -lw 8 -s 128 -g 0,1,2,3,4,5,6,7

# 使用googLenet网络训练cifar10数据集
./classify.py -t second -m googLenet -save 50 -wn ?.pkl -d cifar10 -en -train -test -epochs 10 -lr 0.001 -b 800 -lw 8 -s 128 -g 0,1,2,3,4,5,6,7
