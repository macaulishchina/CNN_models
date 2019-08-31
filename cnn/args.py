import argparse

parser = argparse.ArgumentParser()

# 默认参数列表
parser.add_argument('-g', '--gpuids', default='0', type=str, help='指定cuda设备的id, 多个id用`,`分隔')
parser.add_argument('-m', '--model', default='vgg16', choices=['alexnet', 'vgg16', 'googLenet'], type=str, help='选择baskbone网络')
parser.add_argument('-d', '--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'caltech101', 'caltech256'], type=str, help='输入数据集名称，默认为cifar10')
parser.add_argument("-download", "--download", help="下载数据集", action="store_true")
parser.add_argument('-b', '--batchsize', default=10, type=int, help='Batch Size 大小')
parser.add_argument('-epochs', '--epochs', default='1', type=str, help='指定训练批次, 多个epoch用`,`分隔')
parser.add_argument('-lr', '--learning_rate', default='0.0001', type=str, help='指定训练批次对应学习率, 多个学习率用`,`分隔')
parser.add_argument('-s', '--input_size', default=224, type=int, help='输入图像尺寸,(is,is) -> (h,w)')
parser.add_argument('-t', '--tag', type=str, help='tag')
parser.add_argument('-wn', '--weights_name', default='?.pkl', type=str, help='读取模型的文件名')
parser.add_argument("-train", "--train", help="训练", action="store_true")
parser.add_argument("-test", "--test", help="测试", action="store_true")
parser.add_argument("-only_test", "--only_test", help="只测试", action="store_true")
parser.add_argument('-si', '--save_interval', default=999999999, type=int, help='保存间隔，单位epoch')
parser.add_argument("-p", "--pretrained", help="使用pytorch提供的预训练权重", action="store_true")
parser.add_argument("-en", "--enhancement", help="使用数据增强", action="store_true")
parser.add_argument("-tc", "--tencrop", help="使用tencrop数据增强", action="store_true")
parser.add_argument('-lw', '--load_workers', default=2, type=int, help='数据读取线程数')
parser.add_argument("-shm", "--share_memory", help="使用共享内存", action="store_true")

parser.add_argument_group()
args = parser.parse_args()
