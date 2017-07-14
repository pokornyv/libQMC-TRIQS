# library of functions to be used with the TRIQS cthyb solver
# tested with TRIQS 1.4
# Vladislav Pokorny; 2016-2017; pokornyv@fzu.cz

import scipy as sp
from time import ctime
from itertools import product
from pytriqs.gf.local import *

###########################################################
## general functions ######################################

def OmegaN(n,beta):
	''' calculates the n-th fermionic Matsubara frequency '''
	return (2.0*n+1.0)*sp.pi/beta


def PrintAndWrite(line,fname):
	'''	print the same line to stdout and to file fname '''
	print(line)
	f = open(fname,'a')
	f.write(line+'\n')
	f.close()

###########################################################
## processing the output Green functions ##################

def TailCoeffs(G,bands_T):
	''' reads the tail of GF '''
	Gtail1_D = {}
	Gtail2_D = {}
	Gtail3_D = {}
	Gtail4_D = {}
	for band1,band2 in product(bands_T, repeat=2):
		Gtail1_D[band1,band2] = sp.real(G['0'][band1,band2].tail[1][0][0])
		Gtail2_D[band1,band2] = sp.real(G['0'][band1,band2].tail[2][0][0])
		Gtail3_D[band1,band2] = sp.real(G['0'][band1,band2].tail[3][0][0])
		Gtail4_D[band1,band2] = sp.real(G['0'][band1,band2].tail[4][0][0])
	return [Gtail1_D,Gtail2_D,Gtail3_D,Gtail4_D]


def TotalDensity(G,bands_T,beta,NMats,gtype):
	''' calculates the density from Green function '''
	NBand = len(bands_T)
	N_F = sp.zeros([NBand,NBand])
	for i,j in product(range(NBand), repeat = 2):
		if gtype == 'leg': N_F[i][j] = sp.real(G[bands_T[i],bands_T[j]].total_density())
		elif gtype == 'tau':
			G_iw = GfImFreq(indices = bands_T,beta = beta,n_points = NMats)
			G_iw << Fourier(G)
			N_F[i][j] = sp.real(G_iw[bands_T[i],bands_T[j]].total_density())
		else:              N_F[i][j] = sp.real(G[bands_T[i],bands_T[j]].total_density())
	return N_F

###########################################################
## functions for writing data files #######################

def WriteG_iw(GF,bands_T,beta,NMats,fname,logfname):
	''' writes a Matsubara function to a file '''
	fout = open(fname,'w')
	NBand = len(bands_T)
	G  = GfImFreq(indices = [0],beta = beta,n_points = NMats)
	MatsFreq_F = sp.zeros(2*NMats)
	k = 0
	for iw in G.mesh:		# filling the array of Matsubara frequencies
		MatsFreq_F[k] = sp.imag(iw)
		k += 1
	for i,j in product(range(NBand), repeat = 2):
		G << GF['0'][bands_T[i],bands_T[j]]
		for nw in range(2*NMats):
			fout.write('{0:.12f}\t{1:.12f}\t{2:.12f}\n'\
			.format(float(MatsFreq_F[nw]),float(sp.real(G.data[nw][0][0])),float(sp.imag(G.data[nw][0][0]))))
		fout.write('\n\n')
	fout.close()
	PrintAndWrite('File '+fname+' written.',logfname)


def WriteG_tau(GF,bands_T,beta,NTau,fname,logfname):
	''' writes an imaginary-time function to a file '''
	fout = open(fname,'w')
	NBand = len(bands_T)
	G  = GfImTime(indices = [0],beta = beta,n_points = NTau)
	Times_F = sp.empty(NTau)
	k = 0
	for tau in G.mesh:	# filling the array of imaginary times
		Times_F[k] = sp.real(tau)
		k += 1	
	for i,j in product(range(NBand), repeat = 2):
		G << GF['0'][bands_T[i],bands_T[j]]
		for tau in range(NTau):
			fout.write('{0:.12f}\t{1:.12f}\t{2:.12f}\n'\
			.format(float(Times_F[tau]),float(sp.real(G.data[tau][0][0])),float(sp.imag(G.data[tau][0][0]))))
		fout.write('\n\n')
	fout.close()
	PrintAndWrite('File '+fname+' written.',logfname)


def WriteGleg(GF,bands_T,beta,NLeg,fname,logfname):
	''' writes GF in Legendre prepresentation to file '''
	fout = open(fname,'w')
	NBand = len(bands_T)
	G = GfLegendre(indices = [0],beta = beta,n_points = NLeg)
	Leg_F = sp.array(range(NLeg))
	for i,j in product(range(NBand), repeat = 2):
		G << GF['0'][bands_T[i],bands_T[j]]
		for leg in Leg_F:
			fout.write('{0: 4d}\t{1:.12f}\n'.format(int(leg),float(sp.real(G.data[leg][0][0]))))
		fout.write('\n\n')
	fout.close()
	PrintAndWrite('File '+fname+' written.',logfname)


def WriteMatrix(X_D,bands_T,Xtype,logfname):
	''' writes a dict or matrix with two indices to output in a matrix form 
	input: Xtype = "D" = dict, "M" = matrix '''
	out_text = ''
	NBand = len(bands_T)
	for i in range(NBand):
		for j in range(NBand):
			X = X_D[bands_T[i],bands_T[j]] if Xtype == 'D' else X_D[i][j]
			if sp.imag(X) == 0: out_text += '{0: .6f}\t'.format(float(sp.real(X)))
			else: out_text += '{0: .6f}{1:+0.6f}i\t'.format(float(sp.real(X)),float(sp.imag(X)))
		out_text += '\n'
	if logfname != '': PrintAndWrite(out_text,logfname)
	else:              print out_text


def WriteTail(G,bands_T,logfname):
	''' writes the tail of the non-interacting GF to check for inconsistencies '''
	[Gtail1_D,Gtail2_D,Gtail3_D,Gtail4_D] = TailCoeffs(G,bands_T)
	PrintAndWrite('\n',logfname)	
	PrintAndWrite('G0(iw) tail fit: 1 / iw (-)',logfname)
	WriteMatrix(Gtail1_D,bands_T,'D',logfname)
	PrintAndWrite('G0(iw) tail fit: 1 / iw^2 (-) (local impurity levels)',logfname)
	WriteMatrix(Gtail2_D,bands_T,'D',logfname)
	PrintAndWrite('G0(iw) tail fit: 1 / iw^3 (+)',logfname)
	WriteMatrix(Gtail3_D,bands_T,'D',logfname)
	PrintAndWrite('G0(iw) tail fit: 1 / iw^4 (+)',logfname)
	WriteMatrix(Gtail4_D,bands_T,'D',logfname)


def WriteEig(eig,bands_T,InputParam,logfname):
	''' writes the eigenspectrum of the local Hamiltonian '''
	from string import zfill
	NBand = len(bands_T)
	PrintAndWrite('Hamiltonian structure:',logfname)
	PrintAndWrite('  Hilbert space dimension:  {0: 3d}'.format(int(eig.full_hilbert_space_dim)),logfname)
	PrintAndWrite('  Number of blocks: {0: 3d}'.format(int(eig.n_blocks)),logfname)
	PrintAndWrite('  Energies:',logfname)
	for i in range(eig.n_blocks):
		for j in range(len(eig.energies[i])):
			PrintAndWrite('    {0: 3d}\t{1: 3d}\t{2: .8f}\t'\
			.format(i+1,j+1,eig.energies[i][j])+zfill(bin(eig.fock_states[i][j])[2:],NBand),logfname)
	PrintAndWrite('  Ground state energy: {0: .8f}'.format(float(eig.gs_energy)),logfname)
	PrintAndWrite('  :GS_ENERGY  {0: .6f}\t{1: .8f}'.format(float(InputParam),float(eig.gs_energy)),logfname)


def WriteHisto(po):
	''' writes the histogram of perturbation order into file '''
	[low,high] = po.limits
	N = len(po.data)
	f = open('histo_total.dat','w')
	f.write('# Histogram of total perturbation order.\n# Written on '+ctime()+'\n')
	f.write('# limits: {0: 3d} - {1: 3d},\t'.format(int(po.limits[0]),int(po.limits[1])))
	f.write('number of data points: {0: 3d}\n'.format(int(po.n_data_pts)))	
	orders_F = sp.array(range(N))
	avg_pert = sp.sum(orders_F*po.data)/sum(po.data)
	sd_pert  = sp.sqrt(sp.sum((orders_F-avg_pert)**2*po.data)/sum(po.data))
	skew_pert = sp.sum((orders_F-avg_pert)**3/sd_pert**3*po.data)/sum(po.data)
	kurt_pert = sp.sum((orders_F-avg_pert)**4*po.data)/sum(po.data)/sd_pert**4-3
	f.write('# average pert. order: {0: .5f}, standard deviation: {1: .5f}\n'\
	.format(float(avg_pert),float(sd_pert)))
	f.write('# skew: {0: .5f}, excess kurtosis: {1: .5f}\n'\
	.format(float(skew_pert),float(kurt_pert)))
	for i in orders_F:
		f.write('{0: 3d}\t{1: 3d}\n'.format(int(i),int(po.data[i])))
	f.close()
	return [avg_pert,sd_pert,skew_pert,kurt_pert]


