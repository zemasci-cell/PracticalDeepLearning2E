#
#  file:  esc10_cnn_deep_plot.py
#
#  Make 2D CNN loss and error plots
#
#  RTK, 22-May-2023
#  Last update:  22-May-2023
#
################################################################

import pickle
import matplotlib.pylab as plt
import numpy as np

#  3x3 kernel only
tloss,vloss,terr,verr = pickle.load(open("esc10_cnn_deep_3.pkl","rb"))

#  make the plots
epochs = 16
x = range(1,epochs+1)

plt.plot(x, tloss, marker='o', fillstyle='none', linestyle='solid', linewidth=0.7, color='k', label='training loss')
plt.plot(x, vloss, marker='^', fillstyle='none', linestyle='solid', linewidth=0.7, color='k', label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('deep_3x3_loss.png', dpi=300)
plt.savefig('deep_3x3_loss.eps', dpi=300)
plt.close()

plt.plot(x, terr, marker='o', fillstyle='none', linestyle='solid', linewidth=0.7, color='k', label='training error')
plt.plot(x, verr, marker='^', fillstyle='none', linestyle='solid', linewidth=0.7, color='k', label='validation error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig('deep_3x3_error.png', dpi=300)
plt.savefig('deep_3x3_error.eps', dpi=300)
plt.close()

