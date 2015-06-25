# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import os
import petsc4py
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc

import scipy.sparse as spsp

from pysit.util.wrappers.petsc import PetscWrapper

from pysit import *
from pysit.gallery import marmousi2
from pysit.gallery import marmousi

if __name__ == '__main__':
    # Script to solve the 2D Helmholtz equation with multiple RHS using the Petsc Interface
    # to SuperLU_dist, MUMPS, umfpack, or Pardiso (depending on what is installed on the
    # local PETSC environment

    # changing the OMP_NUM_THREADS variable
    # this is recommended in multicore computers
    os.environ["OMP_NUM_THREADS"] = "4"

    # loading the marmousi model 2, the mini-square
    C, C0, m, d = marmousi(pixel_scale='mini')

    # showing the model in which the Helmholtz equation will be solved
    plt.figure(1)
    vis.plot(C, m)

    # extracting the bound from the domain just built
    xmin = d.x.lbound
    xmax = d.x.rbound
    nx = m.x.n
    zmin = d.z.lbound
    zmax = d.z.rbound

    # Set up shots
    Nshots = 1
    shots = []

    # the depth in which the receivers are located
    zpos = zmin + (1. / 9.) * zmax

    # defining the shots and the receivers (we need to interface to be able to
    # extract the rhs afterwards.
    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers='max',
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # defining the solver
    solver = ConstantDensityHelmholtz(m, model_parameters={'C': C}, spatial_shifted_differences=True,
                                      spatial_accuracy_order=4)
    # setting the parameters of the model
    base_model = solver.ModelParameters(m, {'C': C})

    # building the modelling object
    modelling = FrequencyModeling(solver)

    # frequencies for which the solver will be called
    freqs = [5.0]

    ################################################################################
    # Performing low level solve using PETSc using MUMPS or Superlu_dist

    # setting the base model within the solver
    solver.model_parameters = base_model

    # extracting the Helmholtz matrix
    start = time.time()
    H = solver._build_helmholtz_operator(5.0).tocsr()
    end = time.time()
    print "Time elapsed for the contruction of the Helmholtz operator ", end - start

    # number of deegres of freedom
    ndof = H.shape[1]

    # extracting the rhs from the shot using the solver structure
    rhs = solver.WavefieldVector(solver.mesh, dtype=solver.dtype)
    rhs = solver.build_rhs(solver.mesh.pad_array(shots[0].sources.f(nu=5.0)), rhs_wavefieldvector=rhs)

    # storing the rhs in a numpy array
    b_pysit = rhs.data

    # transforming the Sparse matric in Petsc format
    H_petsc = PETSc.Mat().createAIJ(size=H.shape, csr=(H.indptr, H.indices, H.data))

    # Assembling the matrices for PETSC
    H_petsc.assemblyBegin()
    H_petsc.assemblyEnd()

    # Have to transpose the matrix, because we imported a csc matrix instead of a csr
    # H_petsc.transpose()

    ################################################################################
    # we use the PETSc Krylov solvers interface but we only use the preconditioner.
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(H_petsc)
    ksp.setType('preonly') # setting the solver to only use the preconditioner

    # extracting the preconditioner routine and using MUMPS for it
    pc = ksp.getPC()
    pc.setType('lu') # setting the preconditioner to use an LU factorization

    # setting the library for LU factorization
    pc.setFactorSolverPackage('superlu_dist') # using super_lu as a sparse algorithm
    # pc.setFactorSolverPackage('mumps') # using mumps as a sparse algorithm
    # pc.setFactorSolverPackage('mkl_pardiso') # using the pardiso solver in mkl as a sparse algorithm

    # factorizing the matrix
    start = time.time(); pc.setUp(); end = time.time()
    print "Time elapsed for Helmholtz solve the preconditioner bundled with PETSc ", end - start

    # extracting the factorized matrix
    H_inv = pc.getFactorMatrix()

    # defining the rhs
    b = PETSc.Vec().createSeq(ndof)
    # setting the value of the rhs
    b.setValues(range(0, ndof), b_pysit)

    # defining the solution
    x = PETSc.Vec().createSeq(ndof)

    start = time.time(); H_inv.solve(b, x); end = time.time()
    print "Time elapsed for solving the Helmholtz equation using solve bundled with PETSc ", end - start

    # now solving for multiple rhs
    nrhs = 40
    B = PETSc.Mat().createDense([ndof, nrhs])
    B.setUp()

    for ii in range(0, nrhs, 1):
        B.setValues(range(0, ndof), [ii], b_pysit)

    B.assemblyBegin()
    B.assemblyEnd()

    # defining the solution matrix
    X = PETSc.Mat().createDense([ndof, nrhs])
    X.setUp()

    start = time.time()
    H_inv.matSolve(B, X)
    end = time.time()
    print "Time elapsed for solving the Helmholtz equation with ", nrhs, \
        " rhs using matrix solve bundled with PETSc ", end - start

    err = np.linalg.norm(H*X.getDenseArray() - B.getDenseArray())/np.linalg.norm(B.getDenseArray())

    print "error of the direct solver is given by ", err

    ############################################################################
    # Now using the petsc wrapper to perform the same task

    wrapper = PetscWrapper()
    linear_solver = wrapper.factorize(H, 'mumps', PETSc.COMM_WORLD)


    start = time.time()
    Xwrapper = linear_solver(B.getDenseArray())
    end = time.time()

    print "Time elapsed for solving the Helmholtz equation with ", nrhs, \
        " rhs using Petsc wrapper ", end - start

    err = np.linalg.norm(H*Xwrapper - B.getDenseArray())/np.linalg.norm( B.getDenseArray())

    print "error of the direct solver using the wrapper is given by ", err

