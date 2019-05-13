import sys, os
from importlib import reload
import numpy as np
import time
import datetime
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import glob
from copy import deepcopy
from ics.cobraCharmer import pfi as pfiControl


from bokeh.io import output_notebook, show, export_png,export_svgs, save
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import gray
from bokeh.transform import linear_cmap

def stackMotormapImage(storagePath,repeat):
    arm = 'theta'
    
    marray = ['Reverse', 'Forward']

    for m in marray:
        for nCam in [1,2]:
            for n in range(0,repeat):
                stack_image = np.zeros((2048,2592))   
                filelist=glob.glob(storagePath+'/image/'+f'/{arm}{nCam}{m}{n}N*_0001.fits')

                if len(filelist) > 1:

                    for f in filelist:
                        data = fits.getdata(f)
                        stack_image = data+stack_image
                    fits.writeto(storagePath+f'/product/Cam{nCam}{arm}{m}Stack_{n}.fits',stack_image,overwrite=True)
                else:
                    exit()

def showStackedMortormapImage(storagePath, repeat):
    arm = 'theta'
    
    for n in range(repeat):
        phi_list = glob.glob(storagePath+f'/product/Cam*{arm}*_{n}.fits')
        #print(phi_list)
        a = 1000.0
        
        fig=plt.figure(figsize=(10, 10))
        columns = 1
        rows = len(phi_list)

        ax = []
        i = 0
        for f in phi_list:
            hdu = fits.open(f)
            image = np.log10(a*hdu[0].data+1)/np.log10(a)

            basename = os.path.basename(f)

            xsize = [0,2500]
            ysize = [700, 1100]
            

            ax.append( fig.add_subplot(rows, columns, i+1) )
            if i < len(phi_list)-1:
                ax[-1].get_xaxis().set_ticks([])
            ax[-1].set_title(basename, fontsize = 10)
            plt.imshow(image[ysize[0]:ysize[1],xsize[0]:xsize[1]],cmap='gray')
            i=i+1


        plt.savefig(storagePath+f'/product/{arm}Stacked{n}.png')



def stackOntimeImage(storagePath,repeat):
    arm = 'theta'
    
    tarray = [15, 20, 30, 40, 50]
    marray = ['Reverse', 'Forward']

    for t_ms in tarray:
        for m in marray:
            for nCam in [1,2]:
                for n in range(0,repeat):
                    print(f'{t_ms}, {m}')
                    stack_image = np.zeros((2048,2592))   
                    filelist=glob.glob(storagePath+'/ontimeimage/'+f'/{arm}{nCam}_Ontime{t_ms}{m}{n}N*_0001.fits')

                    for f in filelist:
                        data = fits.getdata(f)
                        stack_image = data+stack_image
                    fits.writeto(storagePath+f'/product/Cam{nCam}_Ontime{t_ms}{arm}{m}Stack_{n}.fits',stack_image,overwrite=True)


def showStackedOntimeImage(storagePath, repeat):
    arm = 'theta'
    tarray = [15,  20, 30, 40, 50]
    
    for t_ms in tarray:
        for n in range(repeat):
            phi_list = glob.glob(storagePath+f'/product/*{t_ms}{arm}*_{n}.fits')

            a = 1000.0
            
            fig=plt.figure(figsize=(10, 10))
            columns = 1
            rows = len(phi_list)

            ax = []
            i = 0
            for f in phi_list:
                hdu = fits.open(f)
                image = np.log10(a*hdu[0].data+1)/np.log10(a)

                basename = os.path.basename(f)

                xsize = [0,2500]
                ysize = [700, 1100]
                

                ax.append( fig.add_subplot(rows, columns, i+1) )
                if i < len(phi_list)-1:
                    ax[-1].get_xaxis().set_ticks([])
                ax[-1].set_title(basename, fontsize = 10)
                plt.imshow(image[ysize[0]:ysize[1],xsize[0]:xsize[1]],cmap='gray')
                i=i+1


            plt.savefig(storagePath+f'/product/{arm}{t_ms}Stacked{n}.png')

def main():
    storagePath = '/arrays/rigel/chyan/20190426/200steps'
    stackMotormapImage(storagePath, 3)
    showStackedMortormapImage(storagePath, 3)
 
 
if __name__ == '__main__':
    main()