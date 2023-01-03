wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
mkdir -p cifar10/train
mkdir -p cifar10/test
python make_cifar10.py
rm -rf cifar-10-batches-py