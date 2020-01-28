import glob
import imageio
from pygifsicle import optimize

TIME_GAP = 0.5
png_dir = '..\\Fig\\GIF_Flow2\\'
filenames = glob.glob(png_dir + '2017*.png') + \
    glob.glob(png_dir + '2018*.png')


images = []
for file_name in filenames:
    images.append(imageio.imread(file_name))
imageio.mimsave('..\\Fig\\baseflow_2017_2018.gif', images, fps=2)


optimize('D:\\OneDrive\\USA_Baseflow\\Fig\\baseflow_2017_2018.gif')
