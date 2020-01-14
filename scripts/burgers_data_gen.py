from phi.flow import *


for scene in Scene.list("~/phi/data/control/forced-burgerclash"):
    scene.remove()

SCENE_COUNT = 10
STEP_COUNT = 32
BATCH_SIZE = 10
VISCOSITY = 0.1 / 32
DOMAIN = Domain([128], box=box[0:1])

print("Generating %d scenes with %d STEP_COUNT each." % (SCENE_COUNT, STEP_COUNT))


def random_initial_state():
    CenteredGrid.sample(0, DOMAIN, batch_size=BATCH_SIZE)


@struct.definition()
class InitialState(AnalyticField):

    def __init__(self):
        AnalyticField.__init__(self, rank=1)

    def sample_at(self, idx, collapse_dimensions=True):
        leftloc = np.random.uniform(0.2, 0.4, BATCH_SIZE)
        leftamp = np.random.uniform(0, 3, BATCH_SIZE)
        leftsig = np.random.uniform(0.05, 0.15, BATCH_SIZE)
        rightloc = np.random.uniform(0.6, 0.8, BATCH_SIZE)
        rightamp = np.random.uniform(-3, 0, BATCH_SIZE)
        rightsig = np.random.uniform(0.05, 0.15, BATCH_SIZE)
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data): return data


@struct.definition()
class GaussianForce(AnalyticField):

    def __init__(self):
        AnalyticField.__init__(self, rank=1)
        self.forceloc = np.random.uniform(0.4, 0.6, BATCH_SIZE)
        self.forceamp = np.random.uniform(-0.05, 0.05, BATCH_SIZE) * 32
        self.forcesig = np.random.uniform(0.1, 0.4, BATCH_SIZE)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.forceamp * np.exp(-0.5 * (idx - self.forceloc) ** 2 / self.forcesig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data): return data

# import pylab


for batch_index in range(SCENE_COUNT // BATCH_SIZE):
    scene = Scene.create('~/phi/data/control/forced-burgerclash', count=BATCH_SIZE)
    print(scene)

    world = World()
    u = u0 = DiffusiveVelocity(DOMAIN, velocity=InitialState(), viscosity=VISCOSITY, batch_size=BATCH_SIZE, name='burgers')
    u = world.add(u, physics=Burgers(diffusion_substeps=4))
    force = world.add(FieldEffect(GaussianForce(), ['velocity']))

    # pylab.plot(u0.velocity.data[0,:,0])

    scene.properties = {"dimensions": DOMAIN.resolution.tolist(), "viscosity": VISCOSITY, "force": force.field.forceamp.tolist()}
    scene.write(world.state, frame=0)
    for frame in range(1, STEP_COUNT + 1):
        world.step(dt=1/32)
        scene.write(world.state, frame=frame)

    # pylab.plot(u.velocity.data[0,:,0])
    # pylab.show()
