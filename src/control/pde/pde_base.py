from phi.tf.flow import struct, isplaceholder, consecutive_frames
from ..hierarchy import ObservationPredictor
import tensorflow as tf
if tf.__version__[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()

class PDE(ObservationPredictor):

    def __init__(self):
        self.fields = {}
        self.scalars = {}

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        raise NotImplementedError(self)

    def placeholder_state(self, world, age):
        raise NotImplementedError(self)

    def target_matching_loss(self, target_state, actual_state):
        raise NotImplementedError(self)

    def total_force_loss(self, states):
        raise NotImplementedError(self)


def property_name(trace): 
    return trace.name


def collect_placeholders_channels(placeholder_states, trace_to_channel=property_name):
    if trace_to_channel is None:
        trace_to_channel = property_name
    placeholders = []
    channels = []

    for i, state in enumerate(placeholder_states):
        if state is not None:
            traces = struct.flatten(state, trace=True)
            for trace in traces:
                if isplaceholder(trace.value):
                    placeholders.append(trace.value)
                    channel = trace_to_channel(trace)
                    channels.append(consecutive_frames(channel, len(placeholder_states))[i])
    return placeholders, tuple(channels)
