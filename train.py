import os

from agent.arguments import parser
from agent.dmc import train


if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    flags.load_model = True
    #flags.actor_device_cpu = True
    assert flags.load_model == True
    assert not flags.parallel
    print(flags.extra_cpu_actor_num)
    train(flags)
