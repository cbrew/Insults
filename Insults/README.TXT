README 
------

Chris Brew (cbrew@acm.org, joshnk at Kaggle)

Data files
==========

Place the .csv file that you want to have results for in Data/Inputs/final.csv.
Currently there is a file there consisting of 1000 of the comments from the 
leaderboard data. I am assuming that the format will be close enough.

Data/Inputs/fulltrain.csv is the concatenation of the two labeled files that
Kaggle provided.

The code
========

insults.py is a single Python file running the whole process. It can be run on
its own, but the best thing to do is to run it through Sumatra, which is designed
to keep simulations and experiments orderly, and ensure that research can be 
reproduced. (see http://packages.python.org/Sumatra/parameter_files.html )

To run through Sumatra, install Sumatra, then:

smt run --executable=/usr/local/bin/python --main=insults.py insults.cfg

or (if you don't want Sumatra to do record-keeping for you) just:

python insults.py insults.cfg

The insults.py script takes lots of options, affecting how the learning is done.
Currently, because Sumatra has rough edges, you specify them in a funky way by
putting them in insults.cfg. 

If you put the following in insults.cfg, you will get a prediction for Data/Inputs/final.csv

commands = [["--competition"]]

What this does is to train on "fulltrain.csv" and test on "final.csv"

show.py is a utility for seeing the learning curve that I used to select a training regime.
score.py spits out some stats.