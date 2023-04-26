# Copyright (c) CAIRI AI Lab. All rights reserved

from openstl.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)
from openstl.api import BaseExperiment
import warnings
import os.path as osp

warnings.filterwarnings('ignore')


try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'SimVP.py') \
        if args.config_file is None else args.config_file
    print("Config path >>>>>>: ", cfg_path)
    config = update_config(config, load_config(cfg_path),
                           exclude_keys=['method', 'batch_size', 'val_batch_size', 'sched'])

    config['test'] = True
    # config['exp_name'] = '14000cleanvids_simvp_batch'

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' testing  ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

    # mse = exp.test()
    inputs, trues, preds = exp.test_hidden()
