from phi.flow import *
from control.pde.burgers import InitialState, GaussianForce


for scene in Scene.list('~/phi/data/control/forced-burgers-clash'):
    scene.remove()

scene_count = 1000  # how many examples to generate (training + validation + test)
step_count = 32  # how many solver steps to perform, equal to (sequence length - 1)
batch_size = 10  # How many examples to generate in parallel
viscosity = 0.003125  # Viscosity constant for Burgers equation
domain = Domain([128], box=box[0:1])  # 1D Grid resolution and physical size
dt = 0.03125  # Time increment per solver step

print("Generating %d scenes with %d STEP_COUNT each." % (scene_count, step_count))


# import pylab


for batch_index in range(scene_count // batch_size):
    scene = Scene.create('~/phi/data/control/forced-burgers-clash', count=batch_size)
    print(scene)

    world = World()
    u = u0 = BurgersVelocity(domain, velocity=InitialState(batch_size), viscosity=viscosity, batch_size=batch_size, name='burgers')
    u = world.add(u, physics=Burgers(diffusion_substeps=4))
    force = world.add(FieldEffect(GaussianForce(batch_size), ['velocity']))

    # pylab.plot(u0.velocity.data[0,:,0])

    scene.properties = {"dimensions": domain.resolution.tolist(), "viscosity": viscosity, "force": force.field.forceamp.tolist()}
    scene.write(world.state, frame=0)
    for frame in range(1, step_count + 1):
        world.step(dt=dt)
        scene.write(world.state, frame=frame)

    # pylab.plot(u.velocity.data[0,:,0])
    # pylab.show()
