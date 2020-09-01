import os

def submit(basedir, m, c, task):
    taskstr = '_'+task if task != 'charge' else ''
    bash_filename = basedir+m+'_'+task+'_'+str(c)+'.sh'
    with open(bash_filename, 'w') as sb:
        sb.write('#!/bin/bash\n')
        sb.write('#SBATCH --nodes=1\n')
        sb.write('#SBATCH --partition=small\n')
        sb.write('#SBATCH --job-name=LegalReasoning\n')
        sb.write('#SBATCH --gres=gpu:1\n')

        sb.write('module load cuda/9.2\n')

        sb.write('#echo $CUDA_VISIBLE_DEVICES\n')
        sb.write('#nvidia-smi\n')
        sb.write('echo $PWD\n')
        sb.write('# run the application\n')
        sb.write('python3 main.py -m '+m+taskstr+' -c '+str(c)+'  > slurm-'+task+str(c)+'model-$SLURM_JOB_ID.out')
        sb.close()

    os.system('sbatch '+bash_filename)

if __name__ == '__main__':
    m='lstmibgan'
    c = [0]
    task = ['law', 'toi']
    # task = ['charge', 'law', 'toi']
    for ci in c:
        for ti in task:
            submit('./artifacts/bash/', m, ci, ti)

    print('submit Complete')




