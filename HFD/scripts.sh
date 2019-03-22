virtualenv --system-site-packages -p python3 ./venv

source ./venv/bin/activate

pip install --upgrade pip
pip list
pip install --upgrade tensorflow
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

echo "Note: Maybe downloading cifar10_data"
python3 generate_cifar10_data.py --data-dir cifar10_data
