# pystrat

Playing around with simple trading strategies and *off-the-shelf* machine-learning models from tensorflow.

## NB: this is a work-in-progress

This project focuses on a ~14.5m record, open-source dataset of daily equity data available from Quandl [here](https://www.quandl.com/data/WIKI/documentation/bulk-download).

I'm going to look at some basic strategies and some basic ML techniques from tensorflow to see if we can improve our strategies.  The goal here is to learn.  An assumption I start out with is that we shouldn't be looking to ML to devise strategies for us (at least not yet), but to use ML to enhance existing strategies with reasonably well-understood characteristics.  

1. [DEN1_datasim](DEN1_datasim.ipynb) - download data, clean & partition it, run some basic sims to get a baseline feel for them.
2. [DEN2_features](DEN2_features.ipynb) - prep the data for use in tensorflow/ML models, break data into various featuresets and see what kinds of results we get.
3. [DEN3_sim](DEN3_sim.ipynb) - return to the strategies with an eye to better understanding their baseline performance using pyfolio's nice tearsheets.

