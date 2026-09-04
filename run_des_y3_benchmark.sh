#!/bin/bash
#SBATCH --account=des_g
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_%j.log

cd /pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev

source ~/cosmosis_init.sh
export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev

echo "==============================================="
echo "Running cpp+python benchmark (fast 0d + slow 3d)..."
echo "==============================================="
cosmosis cosmosis-models/des_y3_cpp_python.ini

echo ""
echo "==============================================="
echo "Running cpp+gpu benchmark (fast 0d + slow 3d)..."
echo "==============================================="
cosmosis cosmosis-models/des_y3_cpp_gpu.ini

echo ""
echo "==============================================="
echo "Benchmarks complete"
echo "==============================================="
