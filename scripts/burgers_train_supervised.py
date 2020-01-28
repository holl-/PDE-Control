from control.pde.burgers import BurgersPDE
from control.control_training import ControlTraining
from phi.flow import show

n = 32

app = ControlTraining(n,
                      BurgersPDE(),
                      datapath='~/phi/data/control/forced-burgers-clash',
                      val_range=range(10),
                      train_range=range(10, 100),
                      trace_to_channel=lambda trace: 'burgers_velocity',
                      obs_loss_frames=[n // 2],
                      trainable_networks=['OP%d' % n],
                      sequence_class=None,
                      batch_size=10,
                      view_size=10,
                      learning_rate=1e-3,
                      dt=1 / 32.)
app.prepare()
app.play(10, callback=app.save_model)
# show(app)
