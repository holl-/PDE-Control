from phi.tf.flow import StateDependency, Physics, ConstantField, FieldEffect, FieldPhysics, ADD, placeholder
from .pde_base import PDE


class ScalarEffectControl(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('scalar', 'scalar', single_state=True),
                                StateDependency('pred', 'next_state_prediction', single_state=True, blocking=True)])

    def step(self, effect, dt=1.0, scalar=None, pred=None):
        v2 = pred.prediction[scalar].data
        v1 = scalar.data
        field = effect.field.with_data(v2-v1)
        return effect.copied_with(field=field, age=effect.age + dt)


class IncrementPDE(PDE):

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        self.scalar = ConstantField(0.0, name='scalar', flags=())
        world.add(self.scalar, FieldPhysics(self.scalar.name))
        self.effect = FieldEffect(ConstantField(0, flags=()), ['scalar'], ADD)
        world.add(self.effect, physics=ScalarEffectControl())

    def placeholder_state(self, world, age):
        plstate = placeholder(world.state[self.scalar].copied_with(age=age).shape)
        return world.state.state_replaced(plstate)

    def target_matching_loss(self, target_state, actual_state):
        return None
        # diff = target_state[self.scalar].data - actual_state[self.scalar].data
        # loss = math.l2_loss(diff)
        # return loss

    def total_force_loss(self, states):
        return None

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        s1, s2 = initial_worldstate[self.scalar], target_worldstate[self.scalar]
        center_age = (s1.age + s2.age) / 2
        new_field = initial_worldstate[self.scalar].copied_with(data=(s1.data + s2.data) * 0.5, flags=(), age=center_age)
        result = initial_worldstate.state_replaced(new_field)
        return result
