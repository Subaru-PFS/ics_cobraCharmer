import numpy as np
import pandas as pd
import os
from scipy import stats

from ics.cobraCharmer import pfi as pfiControl

import xml.etree.cElementTree as ET

from bokeh.io import output_notebook, show, export_png,export_svgs, save
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap

#output_notebook()

def readMotorMap(xml,pid):
    tree = ET.ElementTree(file=xml)
    tree.getroot()
    root=tree.getroot()
    
    j1_fwd_reg=[]
    j1_fwd_stepsize=[]
    for i in root[pid-1][2][0].text.split(',')[2:]:
        if i is not '':
            j1_fwd_reg=np.append(j1_fwd_reg,float(i))

    for i in root[pid-1][2][1].text.split(',')[2:]:
        if i is not '':
            j1_fwd_stepsize=np.append(j1_fwd_stepsize,float(i))

    j1_rev_reg=[]
    j1_rev_stepsize=[]
    for i in root[pid-1][2][2].text.split(',')[2:]:
        if i is not '':
            j1_rev_reg=np.append(j1_rev_reg,float(i))

    for i in root[pid-1][2][3].text.split(',')[2:]:
        if i is not '':
            j1_rev_stepsize=np.append(j1_rev_stepsize,-float(i))


    j2_fwd_reg=[]
    j2_fwd_stepsize=[]
    for i in root[pid-1][2][4].text.split(',')[2:]:
        if i is not '':
            j2_fwd_reg=np.append(j2_fwd_reg,float(i))

    for i in root[pid-1][2][5].text.split(',')[2:]:
        if i is not '':
            j2_fwd_stepsize=np.append(j2_fwd_stepsize,float(i))

    j2_rev_reg=[]
    j2_rev_stepsize=[]
    for i in root[pid-1][2][6].text.split(',')[2:]:
        if i is not '':
            j2_rev_reg=np.append(j2_rev_reg,float(i))

    for i in root[pid-1][2][7].text.split(',')[2:]:
        if i is not '':
            j2_rev_stepsize=np.append(j2_rev_stepsize,-float(i))    
    return j1_fwd_reg,j1_fwd_stepsize,j1_rev_reg,j1_rev_stepsize,\
           j2_fwd_reg,j2_fwd_stepsize,j2_rev_reg,j2_rev_stepsize

def plotJ1OntimeSpeed(GroupIdx, dataFrame, xrange, yrange):

    mapper = Category20[len(GroupIdx)]
    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

    p = figure( tools=TOOLS, x_range=xrange, y_range=yrange,plot_height=500, plot_width=1000)

    colorcode = 0
    for i in GroupIdx:
        legendname="Fiber "+str(i)
        p.line(x=dataFrame['J1onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J1_fwd'][dataFrame['fiberNo'] == i],\
        color=mapper[colorcode],line_width=2,legend=legendname)
        p.circle(x=dataFrame['J1onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J1_fwd'][dataFrame['fiberNo'] == i],radius=0.3,\
            color=mapper[colorcode],fill_color=None)
        p.line(x=dataFrame['J1onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J1_rev'][dataFrame['fiberNo'] == i],color=mapper[colorcode],line_width=2)
        p.circle(x=dataFrame['J1onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J1_rev'][dataFrame['fiberNo'] == i],radius=0.3,\
            color=mapper[colorcode],fill_color=None)

        colorcode = colorcode + 1

    return p

def plotJ2OntimeSpeed(GroupIdx, dataFrame, xrange, yrange):

    mapper = Category20[len(GroupIdx)]
    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

    p = figure( tools=TOOLS, x_range=xrange, y_range=yrange,plot_height=500, plot_width=1000)

    colorcode = 0
    for i in GroupIdx:
        legendname="Fiber "+str(i)

        p.line(x=dataFrame['J2onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J2_fwd'][dataFrame['fiberNo'] == i],\
        color=mapper[colorcode],line_width=2,legend=legendname)
        p.circle(x=dataFrame['J2onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J2_fwd'][dataFrame['fiberNo'] == i],radius=0.3,\
            color=mapper[colorcode],fill_color=None)
        p.line(x=dataFrame['J2onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J2_rev'][dataFrame['fiberNo'] == i],color=mapper[colorcode],line_width=2)
        p.circle(x=dataFrame['J2onTime'][dataFrame['fiberNo'] == i], y=dataFrame['J2_rev'][dataFrame['fiberNo'] == i],radius=0.3,\
            color=mapper[colorcode],fill_color=None)

        colorcode = colorcode + 1

    return p




def main():
    dataPath='/home/pfs/mhs/devel/ics_cobraCharmer/xml/'
    figpath='/Volumes/Disk/Data/MotorMap/'

    #define the broken/good cobras
    brokens = [1, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    goodIdx = np.array(visibles) - 1

    # three non-interfering groups for good cobras
    goodGroupIdx = {}
    for group in range(6):
        goodGroupIdx[group] = goodIdx[goodIdx%6==group] + 1


    dataarray=[]

    filerange = range(25,145,10)
    thetarange = range(25,60,10)
    phirange = range(25,145,10)

    for tms in filerange:
        
        xml2=dataPath+f'motormapOntime{tms}_20181221.xml'
        # Prepare the data path for the work
        # if not (os.path.exists(mappath)):
        #     os.makedirs(mappath)
        pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
        pfi.loadModel(xml2)

        for i in visibles:
            pid=i
            J1onTime=pfi.calibModel.motorOntimeFwd1[i-1]*1000
            J2onTime=pfi.calibModel.motorOntimeFwd2[i-1]*1000

            j1_fwd_reg2,j1_fwd_stepsize2,j1_rev_reg2,j1_rev_stepsize2,\
                    j2_fwd_reg2,j2_fwd_stepsize2,j2_rev_reg2,j2_rev_stepsize2=readMotorMap(xml2,pid)


            a=[pid, J1onTime, J2onTime,np.mean(j1_fwd_stepsize2),np.mean(j1_rev_stepsize2),
                       np.mean(j2_fwd_stepsize2),np.mean(j2_rev_stepsize2)]
            dataarray.append(a)

    data=np.array(dataarray)
    d={'fiberNo': data[:,0],
       'J1onTime': data[:,1], 'J2onTime': data[:,2], 
       'J1_fwd': data[:,3], 'J1_rev': data[:,4],
       'J2_fwd': data[:,5], 'J2_rev': data[:,6]}
    
    df = pd.DataFrame(d)

    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

    p1 = plotJ1OntimeSpeed(goodGroupIdx[0], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])
    p2 = plotJ1OntimeSpeed(goodGroupIdx[1], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])
    p3 = plotJ1OntimeSpeed(goodGroupIdx[2], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])
    p4 = plotJ1OntimeSpeed(goodGroupIdx[3], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])
    p5 = plotJ1OntimeSpeed(goodGroupIdx[4], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])
    p6 = plotJ1OntimeSpeed(goodGroupIdx[5], df, [thetarange[0], thetarange[-1]], [-0.3,0.3])


    x = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    for tms in thetarange:
        x=np.append(x,tms)
        y1=np.append(y1,np.mean(df['J1_fwd'][df['J1onTime']==tms].values))
        y2=np.append(y2,np.mean(df['J1_rev'][df['J1onTime']==tms].values))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y1)

    q = figure(tools=TOOLS, x_range=[thetarange[0]-10, thetarange[-1]+10], y_range=[-1,1],plot_height=500, plot_width=1000)
    q.circle(x=df['J1onTime'], y=df['J1_fwd'],radius=0.3,\
            color='red',fill_color=None)
    q.circle(x=x,y=y1, radius=0.5, color='blue')
    legendtext=f'Y={slope:.8f}X{intercept:.2f}'
    q.line(x=x, y=x*slope+intercept,color='blue',line_width=3, legend=legendtext)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y2)
    #fit=np.polyfit(x, y*100, 1)
    q.circle(x=df['J1onTime'], y=df['J1_rev'],radius=0.3,\
            color='green',fill_color=None)
    q.circle(x=x,y=y2, radius=0.5, color='#3B0F6F')
    legendtext=f'Y={slope:.8f}X+{intercept:.2f}'
    q.line(x=x, y=x*slope+intercept,color='#3B0F6F',line_width=3, legend=legendtext)

    output_file("theta.html")
    save(column(p1,p2,p3,p4,p5,p6,q), filename="theta.html", \
        title='Theta On-time')

#-------------------------------------------
    p1 = plotJ2OntimeSpeed(goodGroupIdx[0], df, [phirange[0], phirange[-1]], [-1,1])
    p2 = plotJ2OntimeSpeed(goodGroupIdx[1], df, [phirange[0], phirange[-1]], [-1,1])
    p3 = plotJ2OntimeSpeed(goodGroupIdx[2], df, [phirange[0], phirange[-1]], [-1,1])
    p4 = plotJ2OntimeSpeed(goodGroupIdx[3], df, [phirange[0], phirange[-1]], [-1,1])
    p5 = plotJ2OntimeSpeed(goodGroupIdx[4], df, [phirange[0], phirange[-1]], [-1,1])
    p6 = plotJ2OntimeSpeed(goodGroupIdx[5], df, [phirange[0], phirange[-1]], [-1,1])

    
    x = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    for tms in phirange:
        #print(np.mean(df['J2_fwd'][df['onTime']==tms].values))
        x=np.append(x,tms)
        y1=np.append(y1,np.median(df['J2_fwd'][df['J2onTime']==tms].values))
        y2=np.append(y2,np.median(df['J2_rev'][df['J2onTime']==tms].values))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:8],y1[:8])
    #fit=np.polyfit(x, y*100, 1)
    #print(fit)

    q = figure(tools=TOOLS, x_range=[phirange[0]-10, phirange[-1]+10], y_range=[-1.5,1.5],plot_height=500, plot_width=1000)
    q.circle(x=df['J2onTime'], y=df['J2_fwd'],radius=0.3,\
            color='red',fill_color=None)
    q.circle(x=x,y=y1, radius=0.5, color='blue')
    legendtext=f'Y={slope:.8f}X{intercept:.2f}'
    q.line(x=x[:8], y=x[:8]*slope+intercept,color='blue',line_width=3, legend=legendtext)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:8],y2[:8])
    #fit=np.polyfit(x, y*100, 1)
    q.circle(x=df['J2onTime'], y=df['J2_rev'],radius=0.3,\
            color='green',fill_color=None)
    q.circle(x=x,y=y2, radius=0.5, color='#3B0F6F')
    legendtext=f'Y={slope:.8f}X+{intercept:.2f}'
    q.line(x=x[:8], y=x[:8]*slope+intercept,color='#3B0F6F',line_width=3, legend=legendtext)

    output_file("phi.html")
    #show(column(p1,p2,p3,p4,p5,p6,q))
    save(column(p1,p2,p3,p4,p5,p6,q), filename="phi.html", \
        title='Phi On-time')




if __name__ == '__main__':
    main()
