from openmm.app import PDBFile
from openmm.app import *
from openmm import *
from openmm.app import CharmmPsfFile
from openmm.unit import *
from sys import stdout
from copy import deepcopy
import openmm.app as app
import os

platform = Platform.getPlatformByName('CUDA')

# Function for heavy atom harmonic restraints
# Ensures proper geometry of hydrogens
def add_backbone_posres(system, positions, atoms, restraint_force):
  force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * kilocalories_per_mole/angstroms**2
  force.addGlobalParameter("k", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N', 'S', 'O'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  posres_sys = deepcopy(system)
  posres_sys.addForce(force)
  return posres_sys

# Function for protein backbone atom restraints
# Ensures orientation of water molecules to the proteins
def add_backbone_posres1(system, positions, atoms, restraint_force):
  force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * kilocalories_per_mole/angstroms**2
  force.addGlobalParameter("k", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  posres_sys1 = deepcopy(system)
  posres_sys1.addForce(force)
  return posres_sys1

# Setup
# PDB file must be in the same directory
pdb = PDBFile('centered_output.pdb')

# Using Amber14 Popular protein force field 
# Using respective Amber 14/Tip3 water model force field
forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
modeller=Modeller(pdb.topology, pdb.positions)

# Ensuring entire protein is protonated
modeller.addHydrogens(forcefield)
# Add missing extra particles to the model that are required by a force field.
modeller.addExtraParticles(forcefield)

# Adds water explicitly
# Provides .15M of NaCl 
modeller.addSolvent(forcefield, ionicStrength=0.15*molar)

# Particle mesh Ewald for the long range electrostatic interactions
# 1 nm cutoff for the direct space interactions
# HBond lengths are fixed
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
# Uses Langevin dynamics with LFMiddle discretization
# Uses a friction coefficient of 1ps^-1
# Uses a time step of 2 fs
integrator = LangevinMiddleIntegrator(10*kelvin, 1/picosecond, 2*femtoseconds)
# Distance tolerance within which constraints are maintained, as a fraction of the constrained distance. The lower, the more accurate
integrator.setConstraintTolerance(0.0001)
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True))

# Minimize
# Run for 8 ps. May be adjusted
print('Minimizing...')
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
simulation.step(4000)

# Begin equilibration step 1: HP 50 kcalmol-1k-2, 100ps at 10 K, restraining heavy atoms incl. water oxygen
# Lasts 100 ps
print('Step 1 equilibration commencing...')
posres_sys = add_backbone_posres(system, modeller.positions, modeller.topology.atoms(), 50)
simulation.step(50000)

# Begin equilibration step 2: HP 50 kcalmol-1k-2, 100ps at 10 K, restraining heavy atoms excl. oxygen
# Lasts 100 ps
print('Step 2 equilibration commencing...')
posres_sys = add_backbone_posres(system, modeller.positions, modeller.topology.atoms(), 0)
posres_sys1 = add_backbone_posres1(system, modeller.positions, modeller.topology.atoms(), 50)
simulation.step(50000)

# Begin equilibration step 3: HP 5 kcalmol-1k-2, 100ps at 10 K, restraining heavy atoms excl. oxygen
# Lasts 100 ps
print('Step 3 equilibration commencing...')
posres_sys1 = add_backbone_posres1(system, modeller.positions, modeller.topology.atoms(), 5)
simulation.step(50000)

# Begin equilibration step 4: HP 0 kcalmol-1k-2, 100ps at 10 K, restraining heavy atoms excl. oxygen
# Lasts 100 ps
print('Step 4 equilibration commencing...')
posres_sys1 = add_backbone_posres1(system, modeller.positions, modeller.topology.atoms(), 0)
simulation.step(50000)

# Begin equilibration step 5: Increasing temperature to 310K
# Lasts ~2000ps total
print('Step 5 equilibration commencing...')
simulation.context.setVelocitiesToTemperature(10*kelvin)
print('Warming up the system...')
T = 10
for i in range(31):
  simulation.step(32258)
  temperature = (T+(i*T))*kelvin 
  integrator.setTemperature(temperature)
simulation.context.setVelocitiesToTemperature(310*kelvin)
print('Temperature of 310K reached')

# NVT minimization and equilibration steps ended
# NPT production steps begin
# Pressure set to 1 atm
print('Begin production steps...')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
system.addForce(MonteCarloBarostat(1.01325*bar, 310*kelvin))
integrator = LangevinMiddleIntegrator(310*kelvin, 1/picosecond, 2*femtoseconds)
integrator.setConstraintTolerance(0.0001)

# Added reporters
# Includes XTC trajectory file to be used to find RMSD/RMSF vs Time
# PDB produces 60 frames, one for every ns
# Production runs for 100 ns
simulation.reporters.append(XTCReporter('Cterm0818.xtc', 1000))
simulation.reporters.append(StateDataReporter('Cterm0818.csv', 1000, time=True, temperature=True, kineticEnergy=True, potentialEnergy=True))
simulation.reporters.append(PDBReporter('Cterm0818.pdb', 500000))
simulation.step(50000000)