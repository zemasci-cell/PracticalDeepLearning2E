#
#  file:  fine-tune_vgg16_fraction.py
#
#  RTK, 03-Dec-2023
#  Last update:  10-Dec-2023
#
################################################################

import os

for fraction in [0.01, 0.03, 0.05, 0.1]:
    # Adam
    cmd = "python3 fine-tune_vgg16.py 128 12 imagenet 3 results/fine-tune_vgg16_fraction_%03d_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16.py 128 12 imagenet 3 results/fine-tune_vgg16_fraction_%03d_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16.py 128 12 none 3 results/fine-tune_vgg16_fraction_%03d_none_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16.py 128 12 none 3 results/fine-tune_vgg16_fraction_%03d_none_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 vgg8.py 128 12 0 0 1 results/vgg8_fraction_%03d_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 vgg8.py 128 12 0 0 1 results/vgg8_fraction_%03d_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)

    #  SGD
    cmd = "python3 fine-tune_vgg16_sgd.py 128 12 imagenet 3 results/fine-tune_vgg16_fraction_sgd_%03d_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16_sgd.py 128 12 imagenet 3 results/fine-tune_vgg16_fraction_sgd_%03d_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16_sgd.py 128 12 none 3 results/fine-tune_vgg16_fraction_sgd_%03d_none_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 fine-tune_vgg16_sgd.py 128 12 none 3 results/fine-tune_vgg16_fraction_sgd_%03d_none_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 vgg8_sgd.py 128 12 0 0 1 results/vgg8_fraction_sgd_%03d_run0 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)
    cmd = "python3 vgg8_sgd.py 128 12 0 0 1 results/vgg8_fraction_sgd_%03d_run1 %0.2f" % (int(fraction*100), fraction)
    os.system(cmd)

