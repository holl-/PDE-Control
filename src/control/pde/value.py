from .pde_base import *


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
        self.scalar = ConstantField('scalar', 0.0)
        world.add(self.scalar)
        self.effect = FieldEffect(ConstantField('0', 0), ['scalar'], ADD)
        world.add(self.effect, physics=ScalarEffectControl())

    def placeholder_state(self, world):
        plstate = placeholder(world.state[self.scalar].shape)
        return world.state.with_replacement(plstate)

    def target_matching_loss(self, target_state, actual_state):
        return None
        # diff = target_state[self.scalar].data - actual_state[self.scalar].data
        # loss = math.l2_loss(diff)
        # return loss

    def total_force_loss(self, states):
        return None

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        v1 = initial_worldstate[self.scalar].data
        v2 = target_worldstate[self.scalar].data
        center_age = 0.5*(initial_worldstate.age+target_worldstate.age)
        new_field = initial_worldstate[self.scalar].with_data((v1 + v2) * 0.5)
        result = initial_worldstate.with_replacement(new_field).copied_with(age=center_age)
        return result
