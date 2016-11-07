# driver for playing with sim & ml

# simple trading strategy simulator

import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix

import numpy as np
from scipy import stats

import sklearn
from sklearn import preprocessing as pp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

import sys
import time

import logging as log
log.basicConfig(level=log.DEBUG)

import glob
import os.path
import pickle

import logging as log
log.basicConfig(level=log.DEBUG)

import random
import pdb

pd.set_option('display.width',500)

import tensorflow as tf
import pyfolio as pf
import pyfolio
import collections
import sim

Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def get_univ(f = 'U.pkl', from_dt='2005-01-01'):
# dev driver    
    P = pickle.load(open(f))
    log.info('loaded <%s>',f)
    P.describe()
    U = P[P.index >= '2005-01-01']
    return U
#    U.describe()
#    _,B = sim.sim(U)
#    #plot NAV
#    B.NAV.plot(title='Equal Weight Everyone')
#    return B


log.info('getting univ...')
U = get_univ()

# load ml data    
fname = 'forsims.pkl'
forsims = pickle.load(open(fname))
log.info('read %s', fname)

src_train = forsims['src_train']
src_vlad = forsims['src_vlad']
Kvlad = forsims['Kvlad']
forsims = None
Kvlad.head()

print Kvlad.shape
print src_vlad.data.shape
print src_train.data.shape


def _fitntestRandomForest( train, vlad, max_nodes=1024, steps=100,
                           model_dir='/tmp/rf') :
    # build fit & test random forest for input
    fsize = len(train.data.columns)
    nclasses = len(train.target.unique())

    hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=nclasses, max_nodes=max_nodes,
        num_classes=nclasses, num_features=fsize)
    classifier = tf.contrib.learn.TensorForestEstimator(hparams)
    
    tdata = train.data.as_matrix().astype(np.float32)
    ttgt = train.target.as_matrix().astype(np.float32)
    vdata = vlad.data.as_matrix().astype(np.float32)
    vtgt = vlad.target.as_matrix().astype(np.float32)

    monitors = [tf.contrib.learn.TensorForestLossMonitor(10, 10)]
    classifier.fit(x=tdata, y=ttgt, steps=steps, monitors=monitors)
    result = classifier.evaluate(x=vdata, y=vtgt)#, steps=np.round(steps/10)

    print('Accuracy: {0:f}'.format(result["accuracy"]))
    return result,classifier

# let's train our model
log.info('training RF model')
src_rf   = _fitntestRandomForest(train=src_train, vlad=src_vlad,
                                 model_dir='/tmp/src_rf',steps=100)

# now let's use our trained model to fit the validation set
vdata = src_vlad.data.as_matrix().astype(np.float32)
vtgt = src_vlad.target.as_matrix().astype(np.float32)
log.info('predicting...')
p=src_rf[1].predict( x=vdata)

log.info('results...')

# how'd it do?
R = pd.DataFrame( {'predicted':p,'actual':vtgt})
R['dist'] = np.abs(R.actual-R.predicted)

# avg distance is meaningful.  a null predictor should get about .88, 
#  so anything below provides some edge
print R.dist.mean()
twos=R.dist[R.dist==2]
len(twos.index)/float(len(R.index))

# ok, let's create a df with date,symbol and prediction which we'll then join onto the simulation dataset
V = pd.DataFrame( {'Date':          Kvlad.Date,
                   'Sym':           Kvlad.Sym,
                   'MLPrediction':  R.predicted })
#Kvlad.head()
print U.shape
print V.shape
V.head()

Uv = U[U.index >= V.Date.min()]
print Uv.shape

Uv.reset_index(inplace=True)
Uv.head()
#V.set_index('Date',inplace=True)

Uml = Uv.merge( V, how='left', on=['Date','Sym'] )
Uml.sort_values(['Date','Sym'],inplace=True)

log.info('built annotated universe.')
Uml.set_index('Date',inplace=True)
Uml.head()



#S,B = sim.sim(Uml,sim_FUN=bestworst_ML,kvargs=kvargs)


def bestworst( U,  cfg, kvargs ) :
    # Buy the prior period's losers and sell the prior period's winners in 
    #  proportion to their over- or under-performance of the equal-weighted market.
    N = len(U.index)
    mktR = U.Return.mean()
    Weight = np.add( U.Return, -mktR ) / (-N)
    # now let's ensure that we spend 100% on each side
    U.Weight = 2 * np.sign(Weight) * (abs(Weight) / sum(abs(Weight)))
    return U


def pyfolio_tf ( Balances ):
    Balances.index=Balances.index.tz_localize('UTC')
    pf.create_full_tear_sheet(Balances.NET_Return)


def bestworst_ML( U,  cfg, kvargs ) :
    """ Buy the prior period's losers and sell the prior period's winners in 
          proportion to their over- or under-performance of the equal-weighted market.
        Then, cross-reference ML's views with this.  The possibilities are:
         - Good: We're buying something predicted to go up
         - Good: We're selling something predicted to go down
         - OK  : We're buying something predicted to stay flat
         - OK  : We're selling something predicted to stay flat
         - Bad : We're buying something predicted to go down
         - Bad : We're selling something predicted to go up

         We leave the *OK*s alone and deallocate from the *Bad*s according to 
         value in kvargs 'realloc' and reallocate to the *Goods*.
    """    
    realloc = kvargs.get('realloc', 0.5)
    N = len(U.index)
    mktR = U.Return.mean()
    Weight = np.add( U.Return, -mktR ) / (-N)
    # now let's ensure that we spend 100% on each side
    U.Weight = 2 * np.sign(Weight) * (abs(Weight) / sum(abs(Weight)))

    # now, let's add-in our ML insights
    # we're going to deallocate from these guys
    bad1 = np.logical_and(U.MLPrediction==0, U.Weight > 0) 
    bad2 = np.logical_and(U.MLPrediction==2, U.Weight < 0)
    bads = np.logical_or(bad1, bad2) 
    lbads = np.logical_and(bads, U.Weight>0)
    sbads = np.logical_and(bads, U.Weight<0)
    
    # and reallocate to these
    good1 = np.logical_and(U.MLPrediction==0, U.Weight < 0)
    good2 = np.logical_and(U.MLPrediction==2, U.Weight > 0)
    goods = np.logical_or( good1, good2 ) 
    lgoods = np.logical_and(goods, U.Weight>0)
    sgoods = np.logical_and(goods, U.Weight<0)
    numlgoods = len(U[lgoods].index)
    numsgoods = len(U[sgoods].index)
    
    # how much weight to add to longs & shorts?
    lwt = U[lbads].Weight.sum() * realloc
    swt = U[sbads].Weight.sum() * realloc

    # let's deallocate from bads
    U.Weight = np.where( bads, U.Weight * (1-realloc),U.Weight)

    # and allocate to goods long & short
    if numlgoods > 0:
        U.Weight = np.where( lgoods, U.Weight + (lwt/numlgoods), U.Weight )
    if numsgoods > 0:
        U.Weight = np.where( sgoods, U.Weight + (swt/numsgoods), U.Weight )

    #    pdb.set_trace()

    return U


    
#Sbw,Bbw = sim.sim(U, sim_FUN=bestworst)
#Bbw.NAV.plot(title="LoMacKinlay")

