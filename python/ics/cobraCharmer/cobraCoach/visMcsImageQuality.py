
import numpy as np
from mpfit import mpfit
import pandas as pd
from sqlalchemy import create_engine
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import mpfit
import concurrent.futures
import os
import glob


def fit_gaussian_and_get_fwhm(args):
    x, y, subimage = args
    p = mpfit2dgaussian(subimage)
    fwhm_x = 2.35 * p[3]
    fwhm_y = 2.35 * p[4]
    return fwhm_x, fwhm_y

def visMcsImageQuality(frameNum):
    
    # Assuming you have the necessary credentials
    db_credentials = "postgresql://pfs@db-ics:5432/opdb"
    engine = create_engine(db_credentials)

    # Use the engine to execute SQL queries
    mcsData = pd.read_sql(f'''
        SELECT DISTINCT 
            spot_id, mcs_center_x_pix, mcs_center_y_pix, mcs_second_moment_x_pix, mcs_second_moment_y_pix
        FROM mcs_data
        WHERE
            mcs_frame_id = {frameNum} and spot_id > 0
        -- limit 10
        ''', engine)
    
    search_pattern = os.path.join('/data/raw', '*', 'mcs', f'*{frameNum}*')
    matching_files = glob.glob(search_pattern, recursive=True)
    file = matching_files[0]

    image = pyfits.open(file)[1].data
    
    xcent = mcsData['mcs_center_x_pix'].values
    ycent = mcsData['mcs_center_y_pix'].values
    
    # Use NumPy array slicing to extract subimages efficiently
    subimages = [
        image[int(np.ceil(y))-8:int(np.ceil(y))+8, int(np.ceil(x))-8:int(np.ceil(x))+8]
        for x, y in zip(xcent, ycent)
    ]
    
    args_list = zip(xcent, ycent, subimages)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(fit_gaussian_and_get_fwhm, args_list))

    # Unpack the results
    fwhm_x, fwhm_y = zip(*results)
    
    xcent = mcsData['mcs_center_x_pix']
    ycent = mcsData['mcs_center_y_pix']


    fig, ax = plt.subplots(1, 2,figsize=(14,6), facecolor="white")
    cm = plt.cm.get_cmap('RdYlBu').reversed() 
    plt.suptitle(f'')
    ax = plt.gcf().get_axes()[0]
    xvalue = np.quantile(fwhm_x, [0,0.25,0.5,0.75,1]) 
    sc=ax.scatter(xcent,ycent,
                        c=fwhm_x,marker='s',vmin=0.8*xvalue[1],vmax=1.2*xvalue[3],cmap=cm)
    ax.set_title(f'X')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = fig.colorbar(sc,cax=cax)


    ax = plt.gcf().get_axes()[1]
    yvalue = np.quantile(fwhm_y, [0,0.25,0.5,0.75,1]) 
    sc=ax.scatter(xcent,ycent,
                        c=fwhm_y,marker='s',vmin=0.8*yvalue[1],vmax=1.2*yvalue[3],cmap=cm)
    ax.set_title(f'Y')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = fig.colorbar(sc,cax=cax)

    return xvalue, yvalue


def gaussian_2d(params, x, y):
    amp, x0, y0, sigma_x, sigma_y, offset = params
    model = amp * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + offset
    return model.flatten()

def fit_gaussian_2d(params, fjac=None, x=None, y=None, data=None):
    model = gaussian_2d(params, x, y)
    status = 0
    return [status, model - data.flatten()]


def mpfit2dgaussian(subimage, visPlot=False, debug=False):
    size = subimage.shape
    
    # Example usage
    # Generate example data with noise
    x = np.arange(0, size[0], 1)
    y = np.arange(0, size[0], 1)
    x, y = np.meshgrid(x, y)

    # Initial guess for parameters: amplitude, x0, y0, sigma_x, sigma_y, offset
    maxi = np.max(subimage)
    offset = np.median(subimage)
    max_index = np.unravel_index(np.argmax(subimage, axis=None), subimage.shape)
    
    initial_params = [maxi, max_index[0], max_index[1], 2, 2, offset]

    fa = {'x': x, 'y': y, 'data': subimage}

    # Fit the 2D Gaussian function to the data
    if debug:
        m = mpfit.mpfit(fit_gaussian_2d, initial_params, functkw=fa, nprint=1)
    else:
        m = mpfit.mpfit(fit_gaussian_2d, initial_params, functkw=fa, nprint=0)

    if m.status > 0:
        if debug: print('Fit successful!')
        best_params = m.params
    else:
        if debug: print('Fit failed. Status:', m.status)
        best_params = initial_params

    #print('Best-fit parameters:', best_params)
    
  

    if visPlot == True: 
        # Plotting
        fit_result = gaussian_2d(m.params, x, y)
        residuals = subimage - fit_result.reshape(16, 16)

        plt.figure(figsize=(15, 5))

        # Original Data
        plt.subplot(1, 3, 1)
        plt.imshow(subimage, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        plt.title('Original Data')
        plt.colorbar()

        # Fitted 2D Gaussian
        plt.subplot(1, 3, 2)
        plt.imshow(fit_result.reshape(16, 16), origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        plt.title('Fitted 2D Gaussian')
        plt.colorbar()

        # Residuals
        plt.subplot(1, 3, 3)
        plt.imshow(residuals, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        plt.title('Residuals')
        plt.colorbar()

        plt.show()
    
    return best_params