# BLAS/ATLAS
sudo apt-get install liblapack-dev libblas-dev libatlas-base-dev gfortran

export LAPACK=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so
export ATLAS=/usr/lib/libatlas.so
export BLAS=/usr/lib/x86_64-linux-gnu/libatlas.so

# INTEL MKL
pip uninstall numpy scipy -y
pip install intel-scipy