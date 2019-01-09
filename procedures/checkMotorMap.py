import numpy as np
import pandas as pd
import os
import random

import xml.etree.cElementTree as ET
from ics.cobraCharmer import pfi as pfiControl


from bokeh.io import output_notebook, show, export_png,export_svgs
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import d3

from bokeh.transform import linear_cmap

#output_notebook()
def extractCalibModel(initXML):
    
    pfi = pfiControl.PFI(fpgaHost='localhost', doConnect=False) #'fpga' for real device.
    pfi.loadModel(initXML)
    
    return pfi.calibModel


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



def main():
    dataPath='/Volumes/Disk/Data/xml/'
    xml1=dataPath+'motormaps_181205.xml'
    figpath='/Volumes/Disk/Data/MotorMap/20190109/'
    
    brokens = [1 , 12, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    #visibles = [2,3,4]

    mapper = d3['Category20'][10]

  
    for i in visibles:
    
        pid=i
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

        p = figure(tools=TOOLS, x_range=[0, 600], y_range=[-0.2,0.2],plot_height=400,
                plot_width=1000,title="Fiber No. "+str(int(pid)))

        p.yaxis.axis_label = "Speed"

        q = figure(tools=TOOLS, x_range=[0, 500], y_range=[-0.3,0.3],plot_height=400,plot_width=545)

        q.xaxis.axis_label = "Degree"
        q.yaxis.axis_label = "Speed"
        colorcode = 0
        #for tms in range(25,90,10):

        xml2=dataPath+f'motormap20190109.xml'
        print(xml2)
        
        model1=extractCalibModel(xml1)
        model2=extractCalibModel(xml2)

        legendname="Ontime = "+str(model2.motorOntimeFwd1[pid-1])

        # Prepare the data path for the work
        #if not (os.path.exists(mappath)):
        #    os.makedirs(mappath)

        

        j1_fwd_reg1,j1_fwd_stepsize1,j1_rev_reg1,j1_rev_stepsize1,\
                j2_fwd_reg1,j2_fwd_stepsize1,j2_rev_reg1,j2_rev_stepsize1=readMotorMap(xml1,pid)
        j1_fwd_reg2,j1_fwd_stepsize2,j1_rev_reg2,j1_rev_stepsize2,\
                j2_fwd_reg2,j2_fwd_stepsize2,j2_rev_reg2,j2_rev_stepsize2=readMotorMap(xml2,pid)
        legendname=f"Avg Speed = {np.mean(j1_fwd_stepsize1):.4f} Ontime = {model1.motorOntimeFwd1[i-1]:.4f}"
        p.line(x=j1_fwd_reg1, y=j1_fwd_stepsize1, color='green', line_width=2, legend=legendname)
        
        legendname=f"Avg Speed = {np.mean(j1_fwd_stepsize2):.4f} Ontime = {model2.motorOntimeFwd1[i-1]:.4f}"
        p.line(x=j1_fwd_reg2, y=j1_fwd_stepsize2, color='red', line_width=2,legend=legendname)
        #p.circle(x=j1_fwd_reg2, y=j1_fwd_stepsize2,radius=1, color=mapper[colorcode],fill_color='white')
        legendname=f"Avg Speed = {np.mean(j1_rev_stepsize1):.4f} Ontime = {model1.motorOntimeRev1[i-1]:.4f}"
        p.line(x=j1_rev_reg1, y=j1_rev_stepsize1, color='green', line_width=2,line_dash="4 4", legend=legendname)
        legendname=f"Avg Speed = {np.mean(j1_rev_stepsize2):.4f} Ontime = {model2.motorOntimeRev1[i-1]:.4f}"
        p.line(x=j1_rev_reg2, y=j1_rev_stepsize2, color='red', line_width=2,line_dash="4 4",legend=legendname)
        #p.circle(x=j1_rev_reg2, y=j1_rev_stepsize2,radius=1, color=mapper[colorcode],fill_color='white')

        legendname=f"Avg Speed = {np.mean(j2_fwd_stepsize1):.4f} Ontime = {model1.motorOntimeFwd2[i-1]:0.4f}"
        q.line(x=j2_fwd_reg1, y=j2_fwd_stepsize1, color='green', line_width=3, legend=legendname)
        legendname=f"Avg Speed = {np.mean(j2_fwd_stepsize2):.4f} Ontime = {model2.motorOntimeFwd2[i-1]:0.4f}"
        q.line(x=j2_fwd_reg2, y=j2_fwd_stepsize2, color='red', line_width=2,legend=legendname)
        #q.circle(x=j2_fwd_reg2, y=j2_fwd_stepsize2,radius=1, color='red', fill_color='white')
        
        legendname=f"Avg Speed = {np.mean(j2_rev_stepsize1):.4f} Ontime = {model1.motorOntimeRev2[i-1]:.4f}"
        q.line(x=j2_rev_reg1, y=j2_rev_stepsize1, color='green', line_width=2,line_dash="4 4", legend=legendname)
        legendname=f"Avg Speed = {np.mean(j2_rev_stepsize2):.4f} Ontime = {model2.motorOntimeRev2[i-1]:.4f}"
        q.line(x=j2_rev_reg2, y=j2_rev_stepsize2, color='red', line_width=2,line_dash="4 4",legend=legendname)
        #q.circle(x=j2_rev_reg2, y=j2_rev_stepsize2,radius=1, color='red', fill_color='white')

        colorcode = colorcode+1

        #show(column(p,q))
        export_png(column(p,q),filename=figpath+"motormap_"+str(int(pid))+".png")



if __name__ == '__main__':
    main()
