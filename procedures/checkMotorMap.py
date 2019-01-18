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

from bokeh.transform import linear_cmap

from bokeh.palettes import Category20

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

def generateMotorMap(baseXML, newXML, figPath, fiberlist = False):
    xml1 = baseXML 
    xml2 = newXML

    model1 = extractCalibModel(xml1)
    model2 = extractCalibModel(xml2)
    
    if fiberlist is not False:
        visibles = fiberlist
    else:
        visibles = range(57)

    for i in visibles:
    
        pid=i
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

        p = figure(tools=TOOLS, x_range=[0, 550], y_range=[-0.2,0.2],plot_height=400,
                plot_width=1000,title="Fiber No. "+str(int(pid)))

        p.yaxis.axis_label = "Speed"

        q = figure(tools=TOOLS, x_range=[0, 300], y_range=[-0.3,0.3],plot_height=400,plot_width=1000)

        q.xaxis.axis_label = "Degree"
        q.yaxis.axis_label = "Speed"

        legendname="Ontime = "+str(model2.motorOntimeFwd1[pid-1])

        # Prepare the data path for the work
        if not (os.path.exists(figPath)):
            os.makedirs(figPath)

        # calculate the limiting index for motor map
        j1limit1 = (360/np.rad2deg(model1.angularSteps[pid-1])).astype(int)-1
        j2limit1 = (180/np.rad2deg(model1.angularSteps[pid-1])).astype(int)-1

        j1limit2 = (360/np.rad2deg(model2.angularSteps[pid-1])).astype(int)-1
        j2limit2 = (180/np.rad2deg(model2.angularSteps[pid-1])).astype(int)-1

        j1_fwd_reg1,j1_fwd_stepsize1,j1_rev_reg1,j1_rev_stepsize1,\
                j2_fwd_reg1,j2_fwd_stepsize1,j2_rev_reg1,j2_rev_stepsize1=readMotorMap(xml1,pid)
        j1_fwd_reg2,j1_fwd_stepsize2,j1_rev_reg2,j1_rev_stepsize2,\
                j2_fwd_reg2,j2_fwd_stepsize2,j2_rev_reg2,j2_rev_stepsize2=readMotorMap(xml2,pid)
        legendname=f"Avg Speed = {np.mean(j1_fwd_stepsize1[:j1limit1]):.4f} Ontime = {model1.motorOntimeFwd1[pid-1]:.4f}"
        p.line(x=j1_fwd_reg1[:j1limit1], y=j1_fwd_stepsize1[:j1limit1], color='green', line_width=2, legend=legendname)
        
        legendname=f"Avg Speed = {np.mean(j1_fwd_stepsize2[:j1limit2]):.4f} Ontime = {model2.motorOntimeFwd1[pid-1]:.4f}"
        p.line(x=j1_fwd_reg2[:j1limit2], y=j1_fwd_stepsize2[:j1limit2], color='red', line_width=2,legend=legendname)
        p.circle(x=j1_fwd_reg2[:j1limit2], y=j1_fwd_stepsize2[:j1limit2],radius=1, color='red',fill_color='white')
        
        legendname=f"Avg Speed = {np.mean(j1_rev_stepsize1[:j1limit1]):.4f} Ontime = {model1.motorOntimeRev1[pid-1]:.4f}"
        p.line(x=j1_rev_reg1[:j1limit1], y=j1_rev_stepsize1[:j1limit1], color='green', line_width=2,line_dash="4 4", legend=legendname)
       
        legendname=f"Avg Speed = {np.mean(j1_rev_stepsize2[:j1limit2]):.4f} Ontime = {model2.motorOntimeRev1[pid-1]:.4f}"
        p.line(x=j1_rev_reg2[:j1limit2], y=j1_rev_stepsize2[:j1limit2], color='red', line_width=2,line_dash="4 4",legend=legendname)
        p.circle(x=j1_rev_reg2[:j1limit2], y=j1_rev_stepsize2[:j1limit2],radius=1, color='red',fill_color='white')

        legendname=f"Avg Speed = {np.mean(j2_fwd_stepsize1[:j2limit1]):.4f} Ontime = {model1.motorOntimeFwd2[pid-1]:0.4f}"
        q.line(x=j2_fwd_reg1[:j2limit1], y=j2_fwd_stepsize1[:j2limit1], color='green', line_width=3, legend=legendname)
        
        legendname=f"Avg Speed = {np.mean(j2_fwd_stepsize2[:j2limit2]):.4f} Ontime = {model2.motorOntimeFwd2[pid-1]:0.4f}"
        q.line(x=j2_fwd_reg2[:j2limit2], y=j2_fwd_stepsize2[:j2limit2], color='red', line_width=2,legend=legendname)
        q.circle(x=j2_fwd_reg2[:j2limit2], y=j2_fwd_stepsize2[:j2limit2],radius=1, color='red', fill_color='white')
        
        legendname=f"Avg Speed = {np.mean(j2_rev_stepsize1[:j2limit1]):.4f} Ontime = {model1.motorOntimeRev2[pid-1]:.4f}"
        q.line(x=j2_rev_reg1[:j2limit1], y=j2_rev_stepsize1[:j2limit1], color='green', line_width=2,line_dash="4 4", legend=legendname)
        
        legendname=f"Avg Speed = {np.mean(j2_rev_stepsize2[:j2limit2]):.4f} Ontime = {model2.motorOntimeRev2[pid-1]:.4f}"
        q.line(x=j2_rev_reg2[:j2limit2], y=j2_rev_stepsize2[:j2limit2], color='red', line_width=2,line_dash="4 4",legend=legendname)
        q.circle(x=j2_rev_reg2[:j2limit2], y=j2_rev_stepsize2[:j2limit2],radius=1, color='red', fill_color='white')


        #show(column(p,q))
        export_png(column(p,q),filename=figPath+"motormap_"+str(int(pid))+".png")


def compareAvgSpeed(baseXML, targetXML, figPath):
    model1 = extractCalibModel(baseXML)
    model2 = extractCalibModel(targetXML)
    

    size1 = len(model1.angularSteps)

    j1fwd_avg1 = np.zeros(size1)
    j1rev_avg1 = np.zeros(size1)
    j2fwd_avg1 = np.zeros(size1)
    j2rev_avg1 = np.zeros(size1)

    for i in range(size1):
        j1fwd_avg1[i] = np.mean(np.rad2deg(model1.angularSteps[i]/model1.S1Pm[i]))
        j1rev_avg1[i] = np.mean(np.rad2deg(model1.angularSteps[i]/model1.S1Nm[i]))
        j2fwd_avg1[i] = np.mean(np.rad2deg(model1.angularSteps[i]/model1.S2Pm[i]))
        j2rev_avg1[i] = np.mean(np.rad2deg(model1.angularSteps[i]/model1.S2Nm[i]))

    size2 = len(model2.angularSteps)

    j1fwd_avg2 = np.zeros(size2)
    j1rev_avg2 = np.zeros(size2)
    j2fwd_avg2 = np.zeros(size2)
    j2rev_avg2 = np.zeros(size2)

    for i in range(size1):
        j1fwd_avg2[i] = np.mean(np.rad2deg(model2.angularSteps[i]/model2.S1Pm[i]))
        j1rev_avg2[i] = np.mean(np.rad2deg(model2.angularSteps[i]/model2.S1Nm[i]))
        j2fwd_avg2[i] = np.mean(np.rad2deg(model2.angularSteps[i]/model2.S2Pm[i]))
        j2rev_avg2[i] = np.mean(np.rad2deg(model2.angularSteps[i]/model2.S2Nm[i]))


    p1=makeHistoPlot(j1fwd_avg1, j1fwd_avg2, 'Theta Fwd', 'Caltech', 'ASIAA')
    p2=makeHistoPlot(j1rev_avg1, j1rev_avg2, 'Theta Rev', 'Caltech', 'ASIAA')
    p3=makeHistoPlot(j2fwd_avg1, j2fwd_avg2, 'Phi Fwd', 'Caltech', 'ASIAA')
    p4=makeHistoPlot(j2rev_avg1, j2rev_avg2, 'Phi Rev', 'Caltech', 'ASIAA')

    show(column(p1,p2,p3,p4))

    #show(p4)
def makeHistoPlot(avg1, avg2, Title, Legend1, Legend2):
    
    hist1, edges1 = np.histogram(avg1, density=True, bins=np.arange(0.0, 0.2, 0.01))
    hist2, edges2 = np.histogram(avg2, density=True, bins=np.arange(0.0, 0.2, 0.01))

    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']
    p = figure(title=Title, tools=TOOLS, background_fill_color="#fafafa")
    p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
           fill_color="navy", line_color="white", alpha=0.3,legend=Legend1)

    p.step(x=edges2[1:-1],y=hist2[0:-1], color='black',legend=Legend2,line_width=2)

    return p


def compareTwoXML():

    dataPath='/Volumes/Disk/Data/xml/'
    xml1=dataPath+'coarse.xml'
    brokens = [1 , 12, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    
    figpath=f'/Volumes/Disk/Data/MotorMap/20190110/'
    xml2=dataPath+f'motormap_20190117.xml'

    generateMotorMap(xml1, xml2, figpath, fiberlist=visibles)


def plotMotorMapFromMutiXML(xmlList, ledgenList, figPath, fiberlist=False):

    if fiberlist is not False:
        visibles = fiberlist
    else:
        visibles = range(57)

    # Prepare the data path for the work
    if not (os.path.exists(figPath)):
            os.makedirs(figPath)
    
    for pid in visibles:
        TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

        p = figure(tools=TOOLS, x_range=[0, 550], y_range=[-0.2,0.2],plot_height=400,
                plot_width=1000,title="Fiber No. "+str(int(pid)))

        p.yaxis.axis_label = "Speed"

        q = figure(tools=TOOLS, x_range=[0, 300], y_range=[-0.3,0.3],plot_height=400,plot_width=1000)

        q.xaxis.axis_label = "Degree"
        q.yaxis.axis_label = "Speed"
        
        mapper = Category20[20]
        colorcode = 0
        for i, xml in enumerate(xmlList):
            model = extractCalibModel(xml)

            j1limit1 = (360/np.rad2deg(model.angularSteps[pid-1])).astype(int)-1
            j2limit1 = (180/np.rad2deg(model.angularSteps[pid-1])).astype(int)-1
            
            j1_fwd_reg1,j1_fwd_stepsize1,j1_rev_reg1,j1_rev_stepsize1,\
                j2_fwd_reg1,j2_fwd_stepsize1,j2_rev_reg1,j2_rev_stepsize1=readMotorMap(xml,pid)

            legendname = ledgenList[i]
            p.line(x=j1_fwd_reg1[:j1limit1], y=j1_fwd_stepsize1[:j1limit1], color=mapper[colorcode], line_width=2, legend=legendname)
            p.line(x=j1_rev_reg1[:j1limit1], y=j1_rev_stepsize1[:j1limit1], color=mapper[colorcode], line_width=2,line_dash="4 4")#, legend=legendname)

            q.line(x=j2_fwd_reg1[:j2limit1], y=j2_fwd_stepsize1[:j2limit1], color=mapper[colorcode], line_width=3)#, legend=legendname)
            q.line(x=j2_rev_reg1[:j2limit1], y=j2_rev_stepsize1[:j2limit1], color=mapper[colorcode], line_width=2,line_dash="4 4")#, legend=legendname)

            colorcode = colorcode+2

        export_png(column(p,q),filename=figPath+"motormap_"+str(int(pid))+".png")


def main():
    dataPath='/Volumes/Disk/Data/xml/'
    xml1=dataPath+'motormaps_181205.xml'
    brokens = [1 , 12, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]

    xml2=dataPath+f'motormap_20190117.xml'

    figpath=f'/Volumes/Disk/Data/MotorMap/20190117/'
    #generateMotorMap(xml1, xml2, figpath, fiberlist=visibles)
    compareAvgSpeed(xml1, xml2, figpath)

if __name__ == '__main__':
    main()