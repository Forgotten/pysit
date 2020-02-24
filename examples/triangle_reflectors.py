# Std import block
import time
import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import triangular_reflector

if __name__ == '__main__':
    # Setup
    frequencySolver = True
    hybrid=False

    nRcv = 80
    # enable Open MP multithread solver
    os.environ["OMP_NUM_THREADS"] = "16"
    
    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (-1.0, 1.0, pmlx, pmlx)
    z_config = (-1.0, 1.0, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 201, 201)

    #   Generate true wave speed
    C, C0, m, d = triangular_reflector(m,reflector_width = [0.1, 0.1], reflector_position =[(-0.3, -0.2), (-0.3, -0.29)])

 
    h = 2*np.pi/nRcv
    xRcv = np.sin(np.linspace(0, 2*np.pi-h, nRcv))   
    zRcv = np.cos(np.linspace(0, 2*np.pi-h, nRcv))  

    radiousSrc = 0.75

    source_list = []
    for x,z in zip(xRcv, zRcv):
        source_list.append(PointSource(m, (radiousSrc*x, radiousSrc*z), 
                           RickerWavelet(10.0), intensity = 1.0))
   
    #2 PointSource objects are defined above. Group them together in a single SourceSet
    source_set = SourceSet(m,source_list)

    radiousRcv = 0.5

    receivers = ReceiverSet(m, [PointReceiver(m, (radiousRcv*x[0], radiousRcv*x[1])) for x in zip(xRcv, zRcv)])

    shots = []
    for source in source_list:
        receiverscopy = copy.deepcopy(receivers)
        shots.append(Shot(source, receiverscopy))


    if frequencySolver:
        solver = ConstantDensityHelmholtz(m, spatial_accuracy_order=6)
        frequencies = [5.0, 10.0, 20.0]

        # Generate synthetic Seismic data
        print('Generating data...')
        base_model = solver.ModelParameters(m,{'C': C})
        tt = time.time()
        generate_seismic_data(shots, solver, base_model, frequencies=frequencies)
        print('Data generation: {0}s'.format(time.time()-tt))

    else:
        # Define and configure the wave solver
        trange = (0.0,3.0)

        solver_time = ConstantDensityAcousticWave(m,
                                              spatial_accuracy_order=6,
                                              kernel_implementation='omp',
                                              trange=trange)
        # Generate synthetic Seismic data
        print('Generating data...')
        base_model = solver_time.ModelParameters(m,{'C': C})
        tt = time.time()
        generate_seismic_data(shots, solver_time, base_model)
        print('Data generation: {0}s'.format(time.time()-tt))

    # Define and configure the objective function
    if hybrid:
        solver = ConstantDensityAcousticWave(m,
                                             spatial_accuracy_order=4,
                                             trange=trange)
        objective = HybridLeastSquares(solver)
    else:

        solver = ConstantDensityHelmholtz(m,
                                          spatial_accuracy_order=4)
        objective = FrequencyLeastSquares(solver)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running Descent...')
    tt = time.time()

    status_configuration = {'value_frequency'           : 1,
                            'residual_frequency'        : 1,
                            'residual_length_frequency' : 1,
                            'objective_frequency'       : 1,
                            'step_frequency'            : 1,
                            'step_length_frequency'     : 1,
                            'gradient_frequency'        : 1,
                            'gradient_length_frequency' : 1,
                            'run_time_frequency'        : 1,
                            'alpha_frequency'           : 1,
                            }
    invalg.max_linesearch_iterations=20

    #loop_configuration=[(60,{'frequencies' : [2.0, 3.5, 5.0]}), (15,{'frequencies' : [6.5, 8.0, 9.5]})] #3 steps at one set of frequencies and 3 at another set

    # loop_configuration=[(40,{'frequencies' : [10.0]}),
    #                     (20,{'frequencies' : [20.0]}), 
    #                     (5,{'frequencies' : [40.0]})]

    loop_configuration=[(20,{'frequencies' : [5.0]}),
                        (10,{'frequencies' : [10.0]}),
                        (10,{'frequencies' : [20.0]})]

    result = invalg(shots, initial_value, loop_configuration, verbose=True, status_configuration=status_configuration)

    print('...run time:  {0}s'.format(time.time()-tt))

    obj_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    plt.figure()
    plt.semilogy(obj_vals)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('decay_triangle.png')

    plt.figure()
    plt.subplot(3,1,1)
    vis.plot(C0, m)
    plt.title('Initial Model')
    plt.subplot(3,1,2)
    vis.plot(C, m)
    plt.title('True Model')
    plt.subplot(3,1,3)
    vis.plot(result.C, m)
    plt.title('Reconstruction')

    fig = plt.gcf()
    fig.set_size_inches(4, 12)

    plt.savefig('reconstruction_triangle.png')
    #plt.show()

