from control.pde.burgers import BurgersPDE
from control.control_training import ControlTraining, StaggeredSequence
from phi.flow import show

n = 2

app = ControlTraining(n,
                      BurgersPDE(),
                      datapath='~/phi/data/control/forced-burgers-clash',
                      val_range=range(10),
                      trace_to_channel=lambda trace: 'burgers_velocity',
                      train_range=range(10, 100),
                      obs_loss_frames=[n // 2],
                      trainable_networks=['OP%d' % n],
                      sequence_class=StaggeredSequence,
                      batch_size=10,
                      view_size=10,
                      learning_rate=1e-3,
                      dt=1 / 32.)
show(app)
