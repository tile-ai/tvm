export TVM_HOME=/root/tilelang
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH


export INSTALL_DIR=/workspace/omniperf_install
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

export PATH=$INSTALL_DIR/2.0.1/bin:$PATH
export PYTHONPATH=$INSTALL_DIR/python-libs:$PYTHONPATH

omniperf profile -n tl_mfma_row_col_16384 -- /opt/conda/envs/py_3.9/bin/python /root/tilelang/amd_scripts/tl_block_gemm.py
omniperf analyze -p ./workloads/tl_mfma_row_col_16384/MI200 --gui
