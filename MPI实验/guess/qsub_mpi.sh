#!/bin/sh
#PBS -N guess_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=2:ppn=8

PROJECT=guess
NP=16
MASTER=master_ubss1
MASTER_DIR=${PBS_O_WORKDIR}

echo "========== MPI Password Guessing Experiment =========="
echo "USER=${USER}"
echo "PROJECT=${PROJECT}"
echo "NP=${NP}"
echo "MASTER_DIR=${MASTER_DIR}"
echo "PBS_NODEFILE=${PBS_NODEFILE}"
echo "======================================================"

echo "[1] Check files on master ..."
ssh ${MASTER} "ls -lh ${MASTER_DIR}/main && ls -ld ${MASTER_DIR}/files"

if [ $? -ne 0 ]; then
    echo "Cannot find main or files on master. Please compile main before qsub."
    exit 1
fi

echo "[2] Copy executable and files to compute nodes ..."
NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    echo "Copy to ${node} ..."
    scp ${MASTER}:${MASTER_DIR}/main ${node}:/home/${USER}/main 1>&2
    scp -r ${MASTER}:${MASTER_DIR}/files ${node}:/home/${USER}/ 1>&2
done

echo "[3] Start MPI program ..."
echo "Machinefile:"
cat $PBS_NODEFILE

/usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE /home/${USER}/main

echo "[4] Copy result files back ..."
scp -r /home/${USER}/files/ ${MASTER}:${MASTER_DIR}/ 2>&1

echo "========== MPI job finished =========="