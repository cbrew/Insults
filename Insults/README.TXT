README 
Chris Brew (cbrew@acm.org)

insults.py is a single Python file running the whole process. It can be run on
its own, but the best thing to do is to run it through Sumatra, which is designed
to keep simulations and experiments orderly, and ensure that research can be 
reproduced.

To train on the leaderboard data, you can run:
	
smt run --executable=/usr/local/bin/python --main=insults.py -- --tune

or (if you don't want Sumatra to do record-keeping for you) just:

python insults.py --tune

The insults.py script takes lots of options, affecting how the learning is done, and so on.
