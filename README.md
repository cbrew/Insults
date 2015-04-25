Code for Kaggle and Impermium's insult detection competition

How to run
==========

Thanks to Abhay Deshmukh <abhay_2685@yahoo.com> for showing me that the original instructions needed to be improved.

The code lives in the Insults directory next to this readme. You need the Anaconda Python distribution to run it
in the way that I recommend. I use Linux and MacOSX, which have all the necessary command-line tools. If you are on
Windows, you will have to adapt the instructions to make sense for your environment.

  cd insults



To run insults.py we need.
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
