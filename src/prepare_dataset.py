from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def main():

    datafolder = '../data/ptbxl/'
    output_folder = '../output/prepdata'

    models = [
        conf_fastai_xresnet1d101
    ]

    experiments = [
        ('exp0', 'all')
    ]


    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, output_folder, models)
        e.prepare()



if __name__ == '__main__':
    main()