import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='warm_v3',
                    help='Experiment id (default: guandan)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')
# unused
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')

# Training settings

# unused for current setting
parser.add_argument('--parallel', action='store_true',
                    help='multi machine parallel training')
parser.add_argument('--host_address', default='tcp://XXX', type=str,
                    help='host address of multi machine parallel training')
parser.add_argument('--rank', default=0, type=int,
                    help='rank of multi machine parallel training')
parser.add_argument('--word_size', default=1, type=int,
                    help='word size of multi machine parallel training')

parser.add_argument('--mix_warm_up', action='store_true',
                    help='mix warm up')
parser.add_argument('--mix_num', default=0.5, type=int,
                    help='mix frac')
parser.add_argument('--no_gpu', action='store_true',
                    help='not use gpu even specified')
parser.add_argument('--rule_generator', action='store_true',
                    help='Use rule as generator')
parser.add_argument('--train_actor', action='store_true',
                    help='Use training device as extra actor device')
parser.add_argument('--train_actor_num', default=4, type=int,
                    help='extra train actor num')
parser.add_argument('--extra_cpu_actor', action='store_true',
                    help='Use CPU as extra actor device')
parser.add_argument('--extra_cpu_actor_num', default=1, type=int,
                    help='extra CPU actor num')
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0,1,2,3', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=3, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='3', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='ccdm_no_rule',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=70, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')
