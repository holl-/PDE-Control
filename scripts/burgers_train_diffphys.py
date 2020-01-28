from control.pde.burgers import BurgersPDE
from control.control_training import ControlTraining, StaggeredSequence
from phi.flow import show

n = 4

checkpoint_dict = {
    'OP2': '/home/holl/phi/model/control-training/sim_000184/checkpoint_00004301',
    'OP4': '/home/holl/phi/model/control-training/sim_000186/checkpoint_00003333',
    'OP8': '/home/holl/phi/model/control-training/sim_000187/checkpoint_00003000',
    'OP16': '/home/holl/phi/model/control-training/sim_000188/checkpoint_00003000',
    'OP32': '/home/holl/phi/model/control-training/sim_000189/checkpoint_00003000',
}

app = ControlTraining(n,
                      BurgersPDE(),
                      datapath='~/phi/data/control/forced-burgers-clash',
                      val_range=range(10),
                      train_range=range(10, 100),
                      trace_to_channel=lambda trace: 'burgers_velocity',
                      obs_loss_frames=[],
                      trainable_networks=['OP%d' % n],
                      checkpoint_dict=checkpoint_dict,
                      sequence_class=StaggeredSequence,
                      batch_size=10,
                      view_size=10,
                      learning_rate=1e-3,
                      dt=1 / 32.)
show(app)
