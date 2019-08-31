export PROJECT_HOME="/home/yidong.hu/study/CNN_models/cnn"
#  使用vgg16网络训练caltech256数据集
./classify.py -t first -d caltech256 -en -shm -train -test -epochs 10 -lr 0.0001 -b 800 -s 224 -g 0,1,2,3,4,5,6,7
