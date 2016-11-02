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

# default simulator cfg dictionary
DEF_SIM_CFG= { 'FrictionInBps': 10,
               'Verbose'      : True,
               'InitBal'      : 1e7,
               'Reinvest'     : True  }

# columns in prepped univ
SIM_COLS = ["Sym","Product","Instrument",
            "Multiplier","Expiry","Strike",
            "Open","High","Low","Close","Volume"]

SIM_COLS_OUT = ["Prev_Weight", "Weight", "Prev_Qty", "Qty",
                "Trade_Qty",  "Trade_Fric", "PNL", "NET_PNL"]

SIM_COL_BALS =[ "NAV","Friction","PNL","NET_PNL", "Longs","Shorts",
                "Long_Dlrs","Short_Dlrs","Num_Trades","Turnover","NET_Return"]

  
def squarem( df, sym='Sym', min_pct=.9 ) :
# sim_squarem solves the common problem in which you have a large table of
#  data grouped by symbols, some of which have missing data.  You want to
#  'square' the data such that any symbol which is missing 'too much' data
#  is expunged and the remaining data is filled appropriately, leaving you
#  with a dataset which has the same # of observations for each symbol.
#
    bysyms = df.groupby(sym).size()
    idx = df.index.unique()
    onumsyms = len(bysyms)

    minlen = int(round(len(idx) * .9 ))
    keep = bysyms[bysyms > minlen]
    u = df[ df[sym].isin(keep.index) ]
    numsyms = len(keep)
    log.info('Got rid of %d/%d symbols',(numsyms-onumsyms),onumsyms)

    u.replace(0,np.nan,inplace=True)
    u.replace([np.inf, -np.inf], np.nan,inplace=True)

    u.sort_index(inplace=True)

    uidx = u.index.unique()
    # groupby and reindex magic
    z = u.groupby(sym).apply(
        lambda x:  x.reindex(uidx).ffill()).reset_index(0,drop=True)

    # badz = z[z.isnull().any(axis=1)]
    # if len(badz.index) > 0 :
    #     badtimes = badz.index.unique().values
    #     z.drop( badtimes, inplace=True )
    #     for dt in badtimes:
    #         log.info('removed %s for NaNs',pd.to_datetime(str(dt)).strftime(
    #             '%Y-%m-%d'))
    return z


# constructs universe appropriate for use with simulator; any additional columns
#  passed-in via ellipsis will be added to table as named
#
def prep_univ( dateTime, symbol, 
               open, high, low, close, volume,
               product, instrument='STK', multiplier=1.0,expiry=None,
               strike=None,adv_days=20,sd_days=20, open2close_returns=True,
               scaleAndCenter=False,  **more_cols) :
    
    U = pd.DataFrame({'Sym': symbol, 
                      'Product' : product, 'Instrument':instrument,
                      'Multiplier': 1.0, 'Expiry': None, 'Strike':None,
                      'Open':open,'High':high, 'Low':low, 'Close':close,
                      'Volume':volume }, index=dateTime )
    U = U[ SIM_COLS ]
    if len(more_cols) > 0:
        U = pd.concat( [U, pd.DataFrame(more_cols)], axis=1 )

    U.reset_index( inplace=True)
    U.sort_values(['Sym','Date'],inplace=True)
    U.Date = pd.to_datetime(U.Date)
    U.set_index('Date',inplace=True)

    if scaleAndCenter :
        log.debug('prep_univ: scaling & centering')
        raw_scaled = U.groupby('Sym').transform(
            lambda x : (x - x.mean())/x.std())
        U = pd.concat([ u.Sym, raw_scaled], axis=1)
    
    # calculate adv, returns, fwd_returns & change in volume
    U['ADV'] = U.groupby('Sym')['Volume'].apply(
        pd.rolling_mean, adv_days, 1).shift()
    U['DeltaV'] = U.groupby('Sym')['Volume'].transform(
        lambda x : np.log(x / x.shift()) ) 
    U['Return'] = U.groupby('Sym')['Close'].transform(
        lambda x : np.log(x / x.shift()) ) 
    U['Fwd_Close'] = U.groupby('Sym')['Close'].shift(-1)
    U['Fwd_Return'] = U.groupby('Sym')['Close'].transform(
        lambda x : np.log(x / x.shift()).shift(-1) ) # fwd.returns
    U['SD'] = U.groupby('Sym')['Return'].apply(
        pd.rolling_std, sd_days, 1).shift()
    
    if open2close_returns:
        U['Fwd_Open'] = U.groupby('Sym')['Open'].shift(-1)
        U['Fwd_COReturn'] = np.divide(np.add( U.Fwd_Open, -U.Close ),U.Close)

    U.ffill(inplace=True)
    U.sort_index(inplace=True)
        
    return U


# simple, default strategy: equal weight universe on daily basis
def eq_wt( U, cfg, kvargs ) :
    #pdb.set_trace()
    U.Weight = 1/float(len(U.index))
    return U

# given today's Universe U and Yesterday's Y, set U's
#  Prev_Weight and Prev_Qty to Y's Weight & Qty
#  TODO: clean-up
def _getprevs( U, Y ) :
    #  TODO: surely there's a cleaner way to do this...
    wts = Y.reset_index()[['Sym','Weight']]
    wts.columns = ['Sym','Prev_Weight']
    pwts = U[['Sym']].merge( wts, on = 'Sym' )['Prev_Weight']
    U.Prev_Weight=pwts.values
    qts = Y.reset_index()[['Sym','Qty']]
    qts.columns = ['Sym','Prev_Qty']
    pqts = U[['Sym']].merge( qts, on = 'Sym' )['Prev_Qty']
    U.Prev_Qty=pqts.values

# functor to run strategy each day and update tbls ...
# TODO: clean-up
def __sim ( U, FUN, cfg, B, kvargs) :
    # run sim to set weights
    U = FUN( U, cfg, kvargs)

    # set prev values for weight & qty...
    Y = kvargs.pop('_Y', None)
    if Y is not None and not np.all(Y.index==U.index):
        _getprevs(U,Y)
        loop = 1 + int(kvargs.pop('_L'))
    else:
        loop = 0
    kvargs['_L'] = loop
    kvargs['_Y'] = U

    bb = B.iloc[loop]
    #  fill-out trade details 
    NAV = bb.NAV
    tospend = NAV/U.Weight
    U.Qty = np.round((NAV*U.Weight) / (U.Multiplier*U.Close))
    U.Trade_Qty  = U.Qty - U.Prev_Qty
    fbps = 1e-4 * cfg['FrictionInBps']
    U.Trade_Fric = U.Trade_Qty * U.Close * U.Multiplier * fbps 
    U.PNL = (U.Fwd_Close - U.Close) * U.Qty * U.Multiplier
    U.NET_PNL = U.PNL - U.Trade_Fric

    # today's balances are based on yesterday's posns...
    longs = U[U.Qty > 0]
    shorts = U[U.Qty < 0]
    trades = U[U.Trade_Qty != 0]
    
    bb.Friction = U.Trade_Fric.sum()
    bb.PNL = U.PNL.sum()
    bb.NET_PNL = U.NET_PNL.sum()
    bb.Longs = len(longs.index)
    bb.Shorts = len(shorts.index)
    bb.Long_Dlrs = (longs.Close * longs.Multiplier * longs.Qty).sum()
    bb.Short_Dlrs = (shorts.Close * shorts.Multiplier * shorts.Qty).sum()
    bb.Num_Trades = len(trades.index)
    bb.Turnover = (trades.Close * trades.Multiplier
                   * trades.Trade_Qty).sum()/NAV
    
    if loop > 0 :
        yb = B.iloc[loop-1]
        ynav = yb.NAV
        tnav = ynav + yb.NET_PNL
        bb.NAV = tnav
        bb.NET_Return = (tnav-ynav)/ynav
        
    B.iloc[loop] = bb

    # pdb.set_trace()
    return U


# simulator...
#
def sim( univ, sim_FUN=eq_wt, cfg=DEF_SIM_CFG.copy(), kvargs={} ) :
    t0 = time.time()
    all_times = univ.index.unique().values

    # prepare writable/output side of universe
    W = pd.DataFrame( columns=SIM_COLS_OUT, index = univ.index).fillna(0.0)
    U = pd.concat( [univ, W], axis=1 )
    
    # create balances table: one per day
    B = pd.DataFrame( columns = SIM_COL_BALS, index = all_times ).fillna(0.0)
    B.NAV = cfg['InitBal']

    # 'daily' loop
    Z = U.groupby(U.index).apply( __sim, FUN=sim_FUN,
                                  cfg=cfg, B=B, kvargs=kvargs )
    
    log.info('ran over %d days and %d rows in %d secs', len(all_times),
             len(U.index),time.time()-t0)
        
    # summarize results a bit more...?
    #ts=xts(B$Net.Return,order.by=B$DateTime)
    
    # return universe and balances
    #list(U=U,B=B, ts=ts)
    return Z, B

def sharpe(Returns) :
    return np.sqrt(252) * np.mean(Returns)/np.std(Returns)

def random_strat( U, cfg, kvargs ) :
    # random portfolio strategy: picks 'num_names' randomly
    nnames = kvargs.get('num_names',10)
    names = random.sample(U.Sym, nnames )
    U.Weight = np.where( U.Sym.isin( names ), 1/float(nnames), 0 )
                 
    return U

def best_strat( U, cfg, kvargs ) :
    # portfolio strategy: picks 'num_names' based on trailing return
    nnames = kvargs.get('num_names',10)
    #pdb.set_trace()
    best = U.sort_values('Return',ascending=False,
                         na_position='last')['Sym'].head(10).values
    U.Weight = np.where( U.Sym.isin( best ), 1/float(nnames), 0 )                 
    return U

def worst_strat( U, cfg, kvargs ) :
    # portfolio strategy: picks 'num_names' based on trailing return
    nnames = kvargs.get('num_names',10)
    #pdb.set_trace()
    worst = U.sort_values('Return',ascending=True,
                          na_position='last')['Sym'].head(10).values
    U.Weight = np.where( U.Sym.isin( worst ), 1/float(nnames), 0 )                 
    return U


# run given strat repeatedly, plotting NAVs and Returning them
def rtest(U,FUN=random_strat, runs=10):
    # run random_strat 'runs' times and plot NAVs
    N = None
    for i in range(runs) :
        _,b = sim( U, sim_FUN=FUN )
        n = pd.DataFrame(b.NAV)
        N = n if N is None else pd.concat([N,n],axis=1)
    N.plot(legend=False)
    return N

# dev driver
def sim_go():
    p = 'z.pkl'
    P = pickle.load(open(p))
    log.info('loaded <%s>',p)
    P.describe()
    atvi1=P[P.Sym=='ATVI']
    atvi1.Close.plot()
    
    Z = squarem(P)
    atvi2=Z[Z.Sym=='ATVI']
    atvi2.Close.plot()

    Ubig = prep_univ( Z.index, Z.Sym, Z.Open, Z.High, Z.Low,
                      Z.Close, Z.Volume, Z.Sym)

    Usmall = Ubig[ Ubig.Sym.isin(['AAPL','MSFT']) ]
    Usmall = Usmall[Usmall.index > '2016-01-01']

    A = Ubig.groupby('Sym').aggregate( 
        lambda x: x.Close.mean() * x.ADV.mean() ).ADV.reset_index()
    tophalf = A[A.ADV > A.ADV.quantile()]
    Umed = Ubig[ Ubig.Sym.isin( tophalf.Sym ) ]
    
    S,B = sim( Usmall )
    S.to_csv('~/s.csv')
    B.to_csv('~/b.csv')
    B.NAV.plot()
    B.Turnover.plot()

    import cProfile
    import pstats
    cProfile.run('sim(Usmall)','sstats')
    p = pstats.Stats('sstats')
    p.strip_dirs().sort_stats('cumtime').print_stats(10)
    Z = squarem(u)

