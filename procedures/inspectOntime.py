import numpy as np
import pandas as pd
import os
from scipy import stats


import xml.etree.cElementTree as ET

from bokeh.io import output_notebook, show, export_png,export_svgs, save
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.models.glyphs import Text
from bokeh.palettes import viridis

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



def main():
    dataPath='/Volumes/Disk/Data/xml/'
    xml1=dataPath+'motormaps_181205.xml'
    figpath='/Volumes/Disk/Data/MotorMap/'

    #define the broken/good cobras
    brokens = [1, 12, 39, 43, 54]
    visibles= [e for e in range(1,58) if e not in brokens]
    badIdx = np.array(brokens) - 1
    goodIdx = np.array(visibles) - 1

    # two groups for two cameras
    cam_split = 26
    group1 = goodIdx[goodIdx <= cam_split]
    group2 = goodIdx[goodIdx > cam_split]

    # three non-interfering groups for good cobras
    goodGroupIdx = {}
    for group in range(6):
        goodGroupIdx[group] = goodIdx[goodIdx%6==group] + 1




    dataarray=[]

    for tms in range(25,140,10):
        
        xml2=dataPath+f'motormapOntime{tms}_20181221.xml'
        print(xml2)
        mappath=figpath+f'{tms}ms'
        # Prepare the data path for the work
        # if not (os.path.exists(mappath)):
        #     os.makedirs(mappath)

        for i in visibles:
        
            pid=i

            j1_fwd_reg1,j1_fwd_stepsize1,j1_rev_reg1,j1_rev_stepsize1,\
                    j2_fwd_reg1,j2_fwd_stepsize1,j2_rev_reg1,j2_rev_stepsize1=readMotorMap(xml1,pid)
            j1_fwd_reg2,j1_fwd_stepsize2,j1_rev_reg2,j1_rev_stepsize2,\
                    j2_fwd_reg2,j2_fwd_stepsize2,j2_rev_reg2,j2_rev_stepsize2=readMotorMap(xml2,pid)


            a=[tms,pid,np.mean(j1_fwd_stepsize2),np.mean(j1_rev_stepsize2),
                       np.mean(j2_fwd_stepsize2),np.mean(j2_rev_stepsize2)]
            
            dataarray.append(a)

    data=np.array(dataarray)
    d={'onTime': data[:,0], 'fiberNo': data[:,1], 
       'J1_fwd': data[:,2], 'J1_rev': data[:,3],
       'J2_fwd': data[:,4], 'J2_rev': data[:,5]}
    
    df = pd.DataFrame(d)


    mapper = viridis(60)
    TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset','hover']

    p1 = figure( tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[0]:
        legendname="Fiber "+str(i)
        p1.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p1.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p1.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p1.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p2 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[1]:
        legendname="Fiber "+str(i)
        p2.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p2.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p2.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p2.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
   
    p3 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[2]:
        legendname="Fiber "+str(i)
        p3.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p3.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p3.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p3.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p4 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[3]:
        legendname="Fiber "+str(i)
        p4.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p4.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p4.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p4.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p5 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[4]:
        legendname="Fiber "+str(i)
        p5.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p5.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p5.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p5.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p6 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-0.3,0.3],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[5]:
        legendname="Fiber "+str(i)
        p6.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p6.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p6.line(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p6.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J1_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
    
    x = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    for tms in range(25,140,10):
        #print(np.mean(df['J1_fwd'][df['onTime']==tms].values))
        x=np.append(x,tms)
        y1=np.append(y1,np.mean(df['J1_fwd'][df['onTime']==tms].values))
        y2=np.append(y2,np.mean(df['J1_rev'][df['onTime']==tms].values))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y1)
    #fit=np.polyfit(x, y*100, 1)
    #print(fit)

    q = figure(tools=TOOLS, x_range=[20, 140], y_range=[-1,1],plot_height=500, plot_width=1000)
    q.circle(x=df['onTime'], y=df['J1_fwd'],radius=0.3,\
            color='red',fill_color=None)
    q.circle(x=x,y=y1, radius=0.5, color='blue')
    legendtext=f'Y={slope:.8f}X{intercept:.2f}'
    q.line(x=x, y=x*slope+intercept,color='blue',line_width=3, legend=legendtext)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y2)
    #fit=np.polyfit(x, y*100, 1)
    q.circle(x=df['onTime'], y=df['J1_rev'],radius=0.3,\
            color='green',fill_color=None)
    q.circle(x=x,y=y2, radius=0.5, color='#3B0F6F')
    legendtext=f'Y={slope:.8f}X+{intercept:.2f}'
    q.line(x=x, y=x*slope+intercept,color='#3B0F6F',line_width=3, legend=legendtext)

    output_file("theta.html")
    #show(column(p1,p2,p3,p4,p5,p6,q))
    save(column(p1,p2,p3,p4,p5,p6,q), filename="theta.html", \
        title='Theta On-time')

#-------------------------------------------

    p1 = figure( tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[0]:
        legendname="Fiber "+str(i)
        p1.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p1.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p1.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p1.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p2 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[1]:
        legendname="Fiber "+str(i)
        p2.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p2.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p2.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p2.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
   
    p3 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[2]:
        legendname="Fiber "+str(i)
        p3.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p3.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p3.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p3.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p4 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[3]:
        legendname="Fiber "+str(i)
        p4.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p4.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p4.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p4.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p5 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[4]:
        legendname="Fiber "+str(i)
        p5.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p5.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p5.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p5.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)

    p6 = figure(tools=TOOLS, x_range=[20, 65], y_range=[-1,1],plot_height=500, plot_width=1000)

    for i in goodGroupIdx[5]:
        legendname="Fiber "+str(i)
        p6.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],\
        color=mapper[i],line_width=1,legend=legendname)
        p6.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_fwd'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
        p6.line(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],color=mapper[i],line_width=1)
        p6.circle(x=df['onTime'][df['fiberNo'] == i], y=df['J2_rev'][df['fiberNo'] == i],radius=0.3,\
            color=mapper[i],fill_color=None)
    
    x = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    for tms in range(25,140,10):
        #print(np.mean(df['J2_fwd'][df['onTime']==tms].values))
        x=np.append(x,tms)
        y1=np.append(y1,np.median(df['J2_fwd'][df['onTime']==tms].values))
        y2=np.append(y2,np.median(df['J2_rev'][df['onTime']==tms].values))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:8],y1[:8])
    #fit=np.polyfit(x, y*100, 1)
    #print(fit)

    q = figure(tools=TOOLS, x_range=[20, 140], y_range=[-1.5,1.5],plot_height=500, plot_width=1000)
    q.circle(x=df['onTime'], y=df['J2_fwd'],radius=0.3,\
            color='red',fill_color=None)
    q.circle(x=x,y=y1, radius=0.5, color='blue')
    legendtext=f'Y={slope:.8f}X{intercept:.2f}'
    q.line(x=x[:8], y=x[:8]*slope+intercept,color='blue',line_width=3, legend=legendtext)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:8],y2[:8])
    #fit=np.polyfit(x, y*100, 1)
    q.circle(x=df['onTime'], y=df['J2_rev'],radius=0.3,\
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
