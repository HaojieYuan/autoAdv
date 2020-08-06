source /home/haojieyuan/.bashrc
ca tf
cuda9
python test_env1.py
PATH=$(getconf PATH)
unset LD_LIBRARY_PATH

source /home/haojieyuan/.bashrc
pytorch
python test_env2.py