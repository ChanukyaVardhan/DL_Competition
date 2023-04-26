from openstl.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)
from openstl.api import BaseExperiment
import os.path as osp
import warnings
import wandb
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

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    print("Config path >>>>>>: ", cfg_path)
    config = update_config(config, load_config(cfg_path),
                           exclude_keys=['method', 'batch_size', 'val_batch_size', 'sched'])

    print("Training for epochs : ", config['epoch'])
    print("Batch sizes : ", config['batch_size'],
          "    ", config['val_batch_size'])

    # set multi-process settings
    setup_multi_processes(config)
    # wandb.init(
    #         entity="dl_competition",
    #         config={"epochs": args.epoch,
    #                 "batch_size": config['batch_size'], "learning_rate": args.lr},
    # )
    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    wandb.finish()
    if rank == 0 and has_nni:
        nni.report_final_result(mse)
