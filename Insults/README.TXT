README 
------

Chris Brew (cbrew@acm.org, joshnk at Kaggle)

How to run
==========

Prerequisites
-------------
To run the script at all we need.
Python 2.7.3 + scikit-learn 0.13-git + ml_metrics 0.1.1 + pandas 0.8.1 + futures 2.2.0
 (+ matplotlib 1.2.1 for plotting)
Create a conda env with everything except ml-metrics
conda create -n insults pandas=0.8.1 scikit-learn=0.13.1 matplotlib=1.2.1 futures=2.2.0
source activate insults
conda install setuptools
Add in ml-metrics
git clone https://github.com/LostProperty/ml-metrics-patched.git
cd ml-metrics-patched/
python setup.py install
cd ..
Run the code
python insults.py --competition
Monitor results (in a separate shell)
cd $INSULTS_DIR
tail -f Logs/final.log  (Ctrl-C to quit when python finishes)
Hardware
--------
I ran on a four year old iMac with 4Gb of memory and dual core Intel processor.
Later also on a 2014 MacBookPro with Yosemite and flash disk.



General approach
=================

Character n-grams plus carefully tuned SGD using elastic net. Character n-grams because
regular linguistic processing is compromised by weird spelling and grammar. SGD because
it works well with sparse features and can be tuned to give good results.

Data files
==========

Place the .csv file that you want to have results for in Data/Inputs/final.csv.

Data/Inputs/fulltrain.csv is the concatenation of the two labeled files that
Kaggle provided.

The code
========

insults.py is a single Python file running the whole process. 

What this does is to train on "fulltrain.csv" and test on "final.csv". It uses a range of values for 
SGD's alpha parameter

show.py is a utility for seeing the learning curve that I used to select a training regime.
score.py spits out some stats for the various submissions.




Submission plan
---------------

Will generate 21 (7 for each penalty type) models using the commands above, then select the five having the highest cross-validation scores.
