from phi.tf.flow import StateDependency, Physics, placeholder, Burgers, Domain, BurgersVelocity, box, math, tf, residual_block_1d
from .pde_base import PDE


VISCOSITY = 0.1 / 32
DOMAIN = Domain([128], box=box[0:1])


class BurgersPDE(PDE):

    def __init__(self):
        PDE.__init__(self)
        self.burgers = None

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        u0 = BurgersVelocity(DOMAIN, viscosity=VISCOSITY, batch_size=world.batch_size, name='burgers')
        self.burgers = world.add(u0, ReplacePhysics())

    def placeholder_state(self, world, age):
        pl_state = placeholder(world.state[self.burgers].shape).copied_with(age=age)
        return world.state.state_replaced(pl_state)

    def target_matching_loss(self, target_state, actual_state):
        diff = target_state[self.burgers].velocity.data - actual_state[self.burgers].velocity.data
        loss = math.l2_loss(diff)
        return loss

    def total_force_loss(self, states):
        return None

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        b1, b2 = initial_worldstate[self.burgers], target_worldstate[self.burgers]
        center_age = (b1.age + b2.age) / 2
        with tf.variable_scope('OP%d' % n):
            predicted_tensor = op_resnet(b1.velocity.data, b2.velocity.data)
        new_field = b1.copied_with(velocity=predicted_tensor, age=center_age)
        result = initial_worldstate.state_replaced(new_field)
        return result


def op_resnet(initial, target, training=True, trainable=True, reuse=tf.AUTO_REUSE):
    y = tf.concat([initial, target], axis=-1)
    downres_padding = sum([2**i for i in range(5)])
    y = tf.pad(y, [[0, 0], [0, downres_padding], [0, 0]])
    resolutions = [y]
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv1d(resolutions[0], filters, 2, strides=2, activation=tf.nn.relu, padding='valid', name='downconv_%d' % i, trainable=trainable, reuse=reuse)
        for j in range(2):
            y = residual_block_1d(y, filters, name='downrb_%d_%d' % (i, j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block_1d(y, nb_channels, name='centerrb_%d' % j, training=training, trainable=trainable, reuse=reuse)

    for i, resolution_data in enumerate(resolutions[1:]):
        y = math.upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-2:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
            y = tf.layers.conv1d(y, 16, 2, 1, activation=tf.nn.relu, padding='valid', name='upconv_%d' % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block_1d(y, nb_channels, 2, name='uprb_%d_%d' % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
            y = tf.layers.conv1d(y, 1, 2, 1, activation=None, padding='valid', name='upconv_%d' % i, trainable=trainable, reuse=reuse)
    return y


class ReplacePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, dependencies=[StateDependency('next_state_prediction', 'next_state_prediction', single_state=True, blocking=True)])

    def step(self, state, dt=1.0, next_state_prediction=None):
        return next_state_prediction.prediction.burgers
