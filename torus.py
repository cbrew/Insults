# torus.py
#
# Game of Life on a torus, purely for fun... well actually
# there were other reasons too.


from scipy.ndimage import convolve
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm





class Board:
	def __init__(self):
		self.filter = np.array([[1,1,1],[1,10,1],[1,1,1]],dtype='float')
		self.board = np.zeros((40,80),dtype=bool)	
		self.add_glider()
		self.add_glider(x=5,y=20)
		self.fig1 = plt.figure()
	
	def animation(self):
		return animation.FuncAnimation(self.fig1, self.update , frames=500,init_func=self.image, interval=50, blit=True, repeat_delay=1000)

	def add_glider(self, x=0, y=0): 
		for i, j in  [(0,1), (1,2), (2,0), (2,1), (2,2)]: self.board[x+i, y+j] = 1.0

	def image(self):
		self.im = plt.imshow(1.0-board.board,cmap=cm.Greys_r)
		return (self.im, )
	def update(self,*y):
		conv = convolve(self.board,self.filter,mode='wrap')
		self.board = (conv == 12) | (conv == 13) | (conv == 3)
		self.im.set_array(1.0-self.board)
		return (self.im, )

board = Board()
ani = board.animation()
plt.show()









