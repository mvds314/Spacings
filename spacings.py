#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:30:45 2017

@author: martin
"""

import numpy as np
import scipy as sp
import scipy.optimize
from scipy.interpolate import interp1d
import sympy
import unittest

import mpmath as mp
mp.extraprec(50)

class interp1d_mpmath(interp1d):
    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = sp._lib._util._asarray_validated(x, check_finite=False, as_inexact=True,
                                             objects_ok=True)
        x_shape = x.shape
        return x.ravel(), x_shape

class LinearCombinationUniformSpacings():
    r"""
    Random variable representing a linear combination of the spacings obtained
    from a uniform distributions

    Given :math:`n-1` draws :math:`X_i` from a uniform distributions. Let
    :math:`X_{(1)}\leq X_{(2)}\leq\ldots\leq X_{(n-1)}` be the order statistics.

    The spacings are defined as:

    .. math::
        S_i=X_{(i)}-X_{(i-1)},
    where :math:`i=1\ldots n` and :math:`X_{n}=1` by convention.

    This class represents the distribution of

    .. math::
        Z=\sum\limits_{i=1}^n a_i S_i.
        
    The class support three backends:
        
    * numpy: for fast computation
    * mpmath: for accurate compuation
    * sympy: to support the case where some the :math:`a_i` are equal.
    
    The moments can be derived from a flat Dirichlet distribution and are given
    by:
    
    * Mean: :math:`\operatorname{E}Z=\frac{1}{n}\sum\limits_{i=1}^n a_i`
    * Variance: :math:`\operatorname{Var}Z=\frac{n-1}{n^2(n+1)}\sum\limits_{i=1}^n a_i^2-\frac{2}{n^2(n+1)}\sum\limits_{i\neq j} a_ia_j`
    * General moments can be computed analytically, but are currently not implemented

    References
    ----------
    [1] Huffer, Lin (2006) - Linear combinations of spacings
    """
    def __init__(self, a,
                 r=None,
                 backend='numpy'):
        """
        Parameters
        ----------
        a : numpy array
            (unique) coefficients of the linear combination
        r : numpy array (default None)
            indicates occurances of each entry of a, only supported for sympy backend
        backend : string
            numpy (default) or mpmath for higher precision
        """
        assert len(np.unique(a))==len(a)
        self.a=np.asanyarray(a, dtype=np.float64)
        self._backend=backend
        assert self._backend in ['numpy', 'mpmath', 'sympy']
        self.r=None
        if r is not None:
            assert backend=='sympy'
            self.r=np.asanyarray(r).astype(np.int)
            assert np.all(self.r>0)
        if self.r is None and backend=='sympy':
            self.r=np.array([1 for aa in self.a], dtype=np.int)
        self._proda_jni_cache_numpy=None
        self._proda_jni_cache_mpmath=None
        self._icdf_grid_x=None
        self._icdf_grid_y=None
        self._icdf_interpolator=None
        self._symbolic_pdf_cache=None
        self._symbolic_sf_cache=None
        self._t_symb=None
    @property
    def N(self):
        return len(self.a) if self.r is None else self.r.sum()
    @property
    def _a_full(self):
        return self.a if self.r is None else np.repeat(self.a,self.r)
    @property
    def mean(self):
        return np.mean(self._a_full)
    @property
    def var(self):
        A=-1*np.ones((self.N,self.N))
        np.fill_diagonal(A,self.N-1)
        return np.dot(self._a_full, A.dot(self._a_full))/self.N**2/(self.N+1)
    @property
    def std(self):
        return np.sqrt(self.var)
    def rvs(self, size=1, random_state=None):
        """
        Returns a random sample
        """
        return sp.stats.dirichlet.rvs(np.ones(self.N),
                                      size=size,
                                      random_state=random_state).dot(self._a_full)
    @property
    def _proda_jni(self):
        if self._backend == 'numpy':
            return self._proda_jni_numpy
        elif self._backend == 'mpmath':
            return self._proda_jni_mpmath
        else:
            raise Exception("Invalid backend")
    @property
    def _proda_jni_numpy(self):
        if self._proda_jni_cache_numpy is None:
            idx=np.arange(self.N)
            self._proda_jni_cache_numpy =[ np.prod(ai-self.a[idx!=i]) for i, ai in enumerate(self.a)]
        return self._proda_jni_cache_numpy
    @property
    def _proda_jni_mpmath(self):
        if self._proda_jni_cache_mpmath is None:
            idx=np.arange(self.N)
            self._proda_jni_cache_mpmath =np.array([ mp.fprod(ai-self.a[idx!=i]) for i, ai in enumerate(self.a)])
        return self._proda_jni_cache_mpmath
    @property
    def _t(self):
        if self._t_symb is None:
            self._t_symb= sympy.symbols('t')
        return self._t_symb
    @property
    def _symbolic_sf(self):
        if self._symbolic_sf_cache is None:
            #define some stuff
            ais = [sympy.Symbol('a%d'%i) for i in range(len(self.a))]
            gamma = sympy.symbols('gamma')
            g=sympy.Max(gamma-self._t,0)**(self.N-1)
            #symbolic computation
            prod=1
            for i, ai in enumerate(ais):
                prod *= (self._t-ai)**self.r[i]
            f = [g.subs(gamma,ai) for i, ai in enumerate(ais)]
            for i,ai in enumerate(ais):
                f[i]*=((self._t-ai)**self.r[i]/prod).subs(self._t,ai)
                f[i]=f[i].diff(ai,self.r[i]-1)
                f[i]=f[i].subs(zip(ais,self.a))/sympy.factorial(self.r[i]-1)
            self._symbolic_sf_cache=sum(f)
        return self._symbolic_sf_cache
    @property
    def _symbolic_pdf(self):
        if self._symbolic_pdf_cache is None:
            self._symbolic_pdf_cache=-self._symbolic_sf.diff(self._t)
        return self._symbolic_pdf_cache
    def pdf(self, t, **kwargs):
        """
        Density function
        """
        t=np.asanyarray(t)
        if t.ndim !=0:
            return np.array([self.pdf(yy, **kwargs) for yy in t])
        if self._backend == 'numpy':
            return self._pdf_numpy(t.item(), **kwargs)
        elif self._backend=='mpmath':
            return self._pdf_mpmath(t.item(), **kwargs)
        elif self._backend=='sympy':
            return self._pdf_sympy(t.item(), **kwargs)
        else:
            raise Exception("Invalid backend")
    def _pdf_numpy(self,t):
        if t<min(self.a) or t>max(self.a) or\
           np.isclose(t, min(self.a), rtol=0) or\
           np.isclose(t, max(self.a), rtol=0):
            return 0
        y=(self.N-1)*np.power(np.maximum(self.a-t,0), self.N-2)
        y=y/self._proda_jni
        return y.sum()
    def _pdf_mpmath(self,t, output='float'):
        if output=='float':
            return np.float64(self._pdf_mpmath(t,output='mpf'))
        if t<min(self.a) or t>max(self.a) or\
           mp.almosteq(t, min(self.a), 1e-8) or\
           mp.almosteq(t, max(self.a), 1e-8):
            return mp.mpf(0)
        y=[(self.N-1)*mp.power(x,self.N-2) for x in np.maximum(self.a-t,0)]
        y=[y1/y2 for y1, y2 in zip(y, self._proda_jni_mpmath)]
        return mp.fsum(y)
    def _pdf_sympy(self,t):
        if t<min(self.a) or t>max(self.a) or\
           np.isclose(t, min(self.a), rtol=0) or\
           np.isclose(t, max(self.a), rtol=0):
            return 0
        return np.float64(self._symbolic_pdf.subs(self._t,t))
    def sf(self,t, **kwargs):
        """
        Survival function
        
        Paramters
        ---------
        output : string (default)
            only used for mpmath, indicates if floats or mpmath floats are returned
        """
        t=np.asanyarray(t)
        if t.ndim !=0:
            return np.array([self.sf(yy, **kwargs) for yy in t])
        if self._backend == 'numpy':
            return self._sf_numpy(t.item(), **kwargs)
        elif self._backend=='mpmath':
            return self._sf_mpmath(t.item(), **kwargs)
        elif self._backend=='sympy':
            return self._sf_sympy(t.item(), **kwargs)
        else:
            raise Exception("Invalid backend")
    def _sf_numpy(self,t):
        if t<min(self.a) or np.isclose(t,min(self.a), rtol=0):
            return 1
        elif  t>max(self.a) or np.isclose(t,max(self.a), rtol=0):
            return 0
        else:
            y=np.power(np.maximum(self.a-t,0), self.N-1)/self._proda_jni_numpy
            return y.sum()
    def _sf_mpmath(self,t, output='float'):
        if output=='float':
            return np.float64(self._sf_mpmath(t,output='mpf'))
        if t<min(self.a) or mp.almosteq(t, min(self.a), 1e-8):
            return mp.mpf(1)
        elif t>max(self.a) or mp.almosteq(t, max(self.a), 1e-8):
            return mp.mpf(0)
        else:
            y=[mp.power(max(yy,0),self.N-1) for yy in self.a-t]
            y=[y1/y2 for y1, y2 in zip(y, self._proda_jni_mpmath)]
            return mp.fsum(y)
    def _sf_sympy(self,t, output='float'):
        if output=='float':
            return np.float64(self._sf_sympy(t, output='mpf'))
        if t<min(self.a) or mp.almosteq(t,min(self.a),1e-8):
            return mp.mpf(1)
        elif t>max(self.a) or mp.almosteq(t,max(self.a),1e-8):
            return mp.mpf(0)
        else:
            return self._symbolic_sf.subs(self._t,t)
    def cdf(self,t, **kwargs):
        """
        Cumulative density function
        
        Paramters
        ---------
        output : string (default)
            only used for mpmath, indicates if floats or mpmath floats are returned
        """
        return 1-self.sf(t, **kwargs)
    def icdf(self,t, **kwargs):
        """
        Inverse cumulative density function.
        
        Can be computed using a solver which is fast on individual evaluations or
        using interpolation. The interpolation calculates the icdf on a grid and
        uses linear interpolation on future evaluations.
        
        Parameters
        ----------
        output : string (default)
            only used for mpmath, indicates if floats or mpmath floats are returned
        method : string
            can be "solver" or "interpolation
        N : int (default 10000)
            number of interpolation points 
        remaining kwargs : keyword arguments
            are passed to solver
        """
        t=np.asanyarray(t)
        if t.ndim !=0:
            return np.array([self.icdf(yy, **kwargs) for yy in t])
        if self._backend == 'numpy':
            return self._icdf_numpy(t.item(), **kwargs)
        elif self._backend in ['mpmath', 'sympy']:
            return self._icdf_mpmath(t.item(), **kwargs)
        else:
            raise Exception("Invalid backend")
    def _icdf_numpy(self, y, method='solver', N=10000):
        if y<0 or y>1:
            return np.nan
        elif np.isclose(y,0, rtol=0):
            return np.min(self.a)
        elif np.isclose(y,1, rtol=0):
            return np.max(self.a)
        if method=='solver':
            f=lambda t : y-1+self._sf_numpy(t)
            return scipy.optimize.bisect(f, np.min(self.a), np.max(self.a))
        elif method=='interpolation':
            if self._icdf_grid_x is None or len(self._icdf_grid_x) !=N:
                self._icdf_grid_y=np.linspace(min(self.a), max(self.a),N)
                self._icdf_grid_x=self.cdf(self._icdf_grid_y)
                self._icdf_interpolator=interp1d(self._icdf_grid_x, self._icdf_grid_y,
                                                 assume_sorted=True, copy=False)
            return self._icdf_interpolator(y).item()
        else:
            raise Exception('Invalid method')
    def _icdf_mpmath(self, y,
                     method='solver',
                     N=10000,
                     output='float',
                     **kwargs):
        if output=='float':
            return np.float64(self._icdf_mpmath(y, output='mpf',
                                                method=method,
                                                N=N, **kwargs))
        if mp.almosteq(y, 0, 1e-8):
            a_min=np.min(self.a)
            if isinstance(a_min, np.int32):
                a_min=int(a_min)
            return mp.mpf(a_min)
        elif mp.almosteq(y, 1, 1e-8):
            a_max=np.max(self.a)
            if isinstance(a_max, np.int32):
                a_max=int(a_max)
            return mp.mpf(a_max)
        elif y<0 or y>1:
            return mp.nan
        if method=='solver':
            f=lambda t : y-1+self.sf(t)
            x0=(np.min(self.a), np.max(self.a))
            kwargs.setdefault('solver','bisect')
            return mp.findroot(f, x0, **kwargs)
        elif method=='interpolation':
            if self._icdf_grid_x is None or len(self._icdf_grid_x) !=N:
                self._icdf_grid_y=mp.linspace(min(self.a), max(self.a),N)
                self._icdf_grid_x=self.cdf(self._icdf_grid_y)
                self._icdf_interpolator=interp1d_mpmath(self._icdf_grid_x, self._icdf_grid_y,
                                                        assume_sorted=True, copy=False,
                                                        fill_value=mp.nan)
            return self._icdf_interpolator(y).item()
        else:
            raise Exception('Invalid method')

class unittests(unittest.TestCase):
    @property
    def a(self):
        return np.array([1,1.5,2,6,3])
    _Znp=None
    @property
    def Znp(self):
        if self._Znp is None:
            self._Znp=LinearCombinationUniformSpacings(self.a, backend='numpy')
        return self._Znp
    _Zmp=None
    @property
    def Zmp(self):
        if self._Zmp is None:
            self._Zmp=LinearCombinationUniformSpacings(self.a, backend='mpmath')
        return self._Zmp
    _Zsp=None
    @property
    def Zsp(self):
        if self._Zsp is None:
            self._Zsp = LinearCombinationUniformSpacings(self.a, backend='sympy')
        return self._Zsp
    _Zsp2=None
    @property
    def Zsp2(self):
        if self._Zsp2 is None:
            self._Zsp2=LinearCombinationUniformSpacings(np.array([1,2,3]),
                                                        r=[2,3,1],
                                                        backend='sympy')
        return self._Zsp2
    @property
    def N(self):
        return len(self.a)
    def test_consistency_pdf(self):
        Znp=self.Znp
        Zmp=self.Zmp
        Zsp=self.Zsp
        self.assertAlmostEqual(Zmp.pdf(3), Znp.pdf(3))
        self.assertAlmostEqual(Zsp.pdf(3), Znp.pdf(3))
        res=[Zmp.pdf(3), Zmp.pdf(4)]
        self.assertTrue(np.allclose(Zmp.pdf([3,mp.mpf(4)]),res))
        self.assertTrue(np.allclose(Znp.pdf([3,4]),res))
        self.assertTrue(np.allclose(Zsp.pdf([3,4]),res))
        self.assertIsInstance(Zmp.pdf([3,4]), np.ndarray)
        self.assertIsInstance(Zsp.pdf([3,4]), np.ndarray)
        self.assertIsInstance(self.Zsp2.pdf([3,4]), np.ndarray)
    def test_consistency_sf(self):
        Znp=self.Znp
        Zmp=self.Zmp
        Zsp=self.Zsp
        self.assertAlmostEqual(Zmp.sf(3), Znp.sf(3))
        self.assertAlmostEqual(Zsp.sf(3), Znp.sf(3))
        res=[Zmp.sf(3), Zmp.sf(4)]
        self.assertTrue(np.allclose(Zmp.sf([mp.mpf(3),4]),res))
        self.assertTrue(np.allclose(Znp.sf([3,4]),res))
        self.assertTrue(np.allclose(Zsp.sf([3,4]),res))
        self.assertIsInstance(Zmp.sf([3,4]), np.ndarray)   
        self.assertIsInstance(Zsp.sf([3,4]), np.ndarray) 
        self.assertIsInstance(self.Zsp2.sf([3,4]), np.ndarray) 
    def test_pdf_cdf_sf(self):
        for Z in [self.Zmp, self.Znp, self.Zsp, self.Zsp2]:
            self.assertAlmostEqual(Z.cdf(np.max(self.a)),1)
            self.assertAlmostEqual(Z.cdf(np.min(self.a)),0)
            self.assertAlmostEqual(Z.sf(np.min(self.a)),1)
            self.assertAlmostEqual(Z.cdf(2*np.max(self.a)),1)
            self.assertAlmostEqual(Z.cdf(-2*np.min(self.a)),0)
            self.assertAlmostEqual(Z.pdf(2*np.max(self.a)),0)
        Zmp=self.Zmp
        self.assertAlmostEqual(Zmp.pdf(mp.mpf(-2*np.min(self.a))),0)
        self.assertAlmostEqual(Zmp.sf(mp.mpf(np.max(self.a))),0)
    def test_cdf_with_sampling(self):
        np.random.seed(100)
        from statsmodels.distributions.empirical_distribution import ECDF
        sample=sp.stats.dirichlet.rvs(np.ones(self.N),100000).dot(self.a)
        ecdf=ECDF(sample)
        for Z in [self.Zmp, self.Znp, self.Zsp]:
            for val in np.linspace(min(self.a), max(self.a), 10):
                self.assertAlmostEqual(ecdf(val), Z.cdf(val), places=2)
        Z=self.Zsp2
        sample=sp.stats.dirichlet.rvs(np.ones(Z.N),100000).dot(Z._a_full)
        ecdf=ECDF(sample)
        for val in np.linspace(min(Z.a), max(Z.a), 10):
            self.assertAlmostEqual(ecdf(val), Z.cdf(val), places=2)
    def test_consistency_pdf_cdf(self):
        for Z in [self.Zmp, self.Znp, self.Zsp, self.Zsp2]:
            for val in np.linspace(min(self.a), max(self.a), 10):
                self.assertAlmostEqual(sp.integrate.quad(lambda x: Z.pdf(x),min(self.a), val)[0],
                                       Z.cdf(val), places=3)
    def test_special_cases_test_cdf_icdf(self):
        for Z in [self.Zmp, self.Znp, self.Zsp, self.Zsp2]:
            self.assertAlmostEqual(Z.cdf(max(Z.a)),1,places=10)
            self.assertAlmostEqual(Z.cdf(min(Z.a)),0,places=10)
            self.assertAlmostEqual(Z.icdf(0),min(Z.a),places=10)
            self.assertAlmostEqual(Z.icdf(1),max(Z.a),places=10)
    def test_consistency_cdf_icdf(self):
        for Z in [self.Zmp, self.Znp, self.Zsp, self.Zsp2]:
            for val in np.linspace(min(Z.a), max(Z.a), 25):
                self.assertAlmostEqual(Z.icdf(Z.cdf(val)),val, places=10)
                self.assertAlmostEqual(Z.icdf(Z.cdf(val),
                                              method='interpolation',
                                              N=1000),val,
                                       places=4)
    def test_moments_with_sampling(self):
        np.random.seed(100)
        sample=sp.stats.dirichlet.rvs(np.ones(self.N),10000000).dot(self.a)
        for Z in [self.Zmp, self.Znp, self.Zsp]:
            self.assertAlmostEqual(Z.mean, np.mean(sample), places=3)
            self.assertAlmostEqual(Z.var, np.var(sample), places=3)
            self.assertAlmostEqual(Z.std, np.std(sample), places=3)
        Z=self.Zsp2
        sample=sp.stats.dirichlet.rvs(np.ones(Z.N),10000000).dot(Z._a_full)
        self.assertAlmostEqual(Z.mean, np.mean(sample), places=3)
        self.assertAlmostEqual(Z.var, np.var(sample), places=3)
        self.assertAlmostEqual(Z.std, np.std(sample), places=3)
    def test_rvs(self):
        for Z in [self.Zmp, self.Znp, self.Zsp, self.Zsp2]:
            shape=[3,4]
            sample=Z.rvs(size=shape)
            self.assertListEqual(list(sample.shape),shape)
            np.random.seed(10)
            sample1=sp.stats.dirichlet.rvs(np.ones(Z.N),size=(8,6,3)).dot(Z._a_full)
            sample2=Z.rvs(size=(8,6,3), random_state=10)
            self.assertTrue(np.allclose(sample1,sample2))

if __name__ == '__main__':
    run_unit_tests=True
    test_only = list() # if list is empty then test all
    #test_only.append('test_consistency_cdf_icdf')
    if run_unit_tests:
        if len(test_only) > 0:
            suite = unittest.TestSuite()
            for ut in test_only:
                suite.addTest(unittests(ut))
            unittest.TextTestRunner().run(suite)
        else:
            #unittest.main()
            suite = unittest.TestLoader().loadTestsFromTestCase(unittests)
            unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        import matplotlib.pyplot as plt
        plt.close('all')
        #a=np.array([1,1.5,2,6,3])
        a=np.array([1,1.5,1.51,6,3])
        Zmp=LinearCombinationUniformSpacings(a, backend='mpmath')
        Znp=LinearCombinationUniformSpacings(a, backend='numpy')
        Zsp=LinearCombinationUniformSpacings(a, backend='sympy')
        #print some values
        print(Zmp.pdf(3))
        print(Znp.pdf(3))
        print(Zsp.pdf(3))

        #make some plots
        x=np.linspace(np.min(a),np.max(a),200)
        plt.plot(x,Znp.pdf(x))
        plt.plot(x,Znp.cdf(x))
        plt.figure()
        plt.plot(np.linspace(0,1,1000), Znp.icdf(np.linspace(0,1,1000)))
        plt.plot(np.linspace(0,1,1000), Zmp.icdf(np.linspace(0,1,1000)))
        plt.plot(np.linspace(0,1,1000), Zsp.icdf(np.linspace(0,1,1000),
                                               method='interpolation'))
        #show added value of the extra precision of mpmath
        print(Zmp.icdf(mp.mpf('0.999999991'), method='solver'))
        print(Zmp.icdf(mp.mpf('0.999999991'), method='interpolation'))
        print(Znp.icdf(0.999999991, method = 'solver'))
        print(Znp.icdf(0.999999991, method = 'interpolation'))