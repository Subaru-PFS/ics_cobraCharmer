

import datetime
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import os
import pathlib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image

def validate_array_sizes(*arrays):
    """Check if all input arrays have the same length."""
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Arrays have different sizes: {lengths}")
    
def get_total_iteration(pfsVisit):
    #pfsVisitID = visDianosticPlot.findVisit(runDir)
    #pfsDesignID = visDianosticPlot.findDesignFromVisit(pfsVisitID)
    conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'") 
    engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)


    Data = pd.read_sql(f'''
            SELECT DISTINCT pfs_visit_id,iteration FROM public.cobra_match
            WHERE
            cobra_match.pfs_visit_id = %(pfsVisit)s
            ''', engine, params={'pfsVisit': pfsVisit})
    
    return np.max(Data['iteration'].values)

def get_input_iteration(pfsVisit, debug=False):
    directory_path = '/data/logs/actors/fps/'
    search_string = f'{pfsVisit},Preparing'

    fpsLogFile = find_log_file_with_string(directory_path, search_string)
    if debug: print(fpsLogFile)
    with open(fpsLogFile, 'r') as file:
        log_lines = file.readlines()

    for line in log_lines:
        if f'frameId={pfsVisit}00 doCentroid doFibreID with cmd=Command' in line:
            match = re.search(r'KEY\(iteration\)=\[Int\((\d+)\)\]', line)
            if match:
                return int(match.group(1))
            else:
                return None
            
def get_subframe_maxium(pfsVisit, mcsLogFile=None, debug=False):

    if mcsLogFile is None:
        directory_path = '/data/logs/actors/mcs/'
        search_string = f'frameId={pfsVisit}'

        mcsLogFile = find_log_file_with_string(directory_path, search_string)

    # Read the file and extract the contents
    with open(mcsLogFile, 'r') as file:
        file_contents = file.read()

    # Use regular expression to find all occurrences of the pattern
    pattern = re.compile(fr'{re.escape(str(pfsVisit))}(\d{{2}})')
    matches = pattern.findall(file_contents)

    if debug:
        print(pattern,matches)
    # If matches are found, convert them to integers and find the maximum
    if matches:
        max_value = max(map(int, matches[-1]))
        if max_value > 12:
            print("Max iteration is more than 12")
        #print("Maximum value of XX:", max_value)
    else:
        print("No matches found.")

    return max_value

def find_log_file_with_string(directory, search_string):
    log_files = glob.glob(directory + '/*.log')  # Update the file extension if necessary

    for log_file in log_files:
        with open(log_file, 'r') as file:
            log_content = file.read()
            if search_string in log_content:
                return log_file

    return None

def pngs_to_pdf(png_files, pdf_file):
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)

    # A list to store the Image objects
    images = []

    # Convert each PNG file to an Image object and add it to the images list
    for png_file in png_files:
        img = Image(png_file)
        img.drawHeight = 400  # Set the height of the image on the page (adjust as needed)
        img.drawWidth = 500   # Set the width of the image on the page (adjust as needed)
        images.append(img)

    # Build the PDF document with each image on a separate page
    doc.build(images)


def parse_fps_log(visit, debug= False):
    directory_path = '/data/logs/actors/fps/'
    search_string = f'{visit},Preparing'

    fpsLogFile = find_log_file_with_string(directory_path, search_string)
    if debug: print(fpsLogFile)
    
    # Read the log file and extract the time stamps
    with open(fpsLogFile, 'r') as file:
        log_lines = file.readlines()
    start_analysis = False
    
    fpsDone = []
    fpsStart = []
    mcsStart = []
    cobraStart = []
    mcsDone = []
    linkDone = []
    homeCobra = []
    
    total_iteration = get_total_iteration(visit)
    
    goHome = False
    
    # Parse the time stamps and calculate the time spent
    for line in log_lines:
        if not start_analysis:
            if f'{visit},Preparing' in line:
                start_analysis = True
                fpsStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line,start_analysis)
        else:
            if 'new cmd: moveToPfsDesign' in line:
                pass
                #if debug: print(line)
            elif 'Checking passed homed argument = True' in line:
                goHome = True
            elif 'home cobras'in line:
                homeCobra.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'calling mcs expose object' in line:
                mcsStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'Time for exposure' in line:
                mcsDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'Getting positions from DB' in line:
                linkDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'Total detected spots' in line:
                cobraStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'Returing result from matching' in line:
                cobraStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            elif 'cobras did not finish' in line:
                cobraDone = datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ')
                if debug: print(line)
            elif 'We are at design position' in line:
                start_analysis = False
                if debug: print(line)
            elif 'pfsConfig updated successfully' in line:
                start_analysis = False
                fpsDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            #elif total_iteration+1 == len(mcsStart):
            #    start_analysis = False
    tempStart = []
    #print(homeCobra)
    if len(homeCobra) > 1:
        tempStart.append(homeCobra[-1])
    else: 
        tempStart = homeCobra
    for ele in cobraStart:
        tempStart.append(ele)
        #print(tempStart)
    cobrsStart = tempStart

    # Sometimes, there will be file writing error when saving pfsconfig.  If 
    if len(fpsDone) == 0:
        fpsDone.append(cobraDone)
    #print(fpsDone,len(fpsStart))

    return fpsStart, mcsStart, mcsDone, linkDone, tempStart[:-1], fpsDone

def visFPSTimeBar(visit, figName = None):
    fpsStart, mcsStart, mcsDone, linkDone, cobraStart, fpsDone = parse_fps_log(visit)
    
    duration = []
    for i in range(len(cobraStart)):
        duration.append(np.array([
            (mcsStart[i] - cobraStart[i]).total_seconds(), 
            (mcsDone[i] - mcsStart[i]).total_seconds(),  
            (linkDone[i] - mcsDone[i]).total_seconds(),
        ]))

    fpsOps =(cobraStart[0]-fpsStart[0]).total_seconds()

    duration = np.array(duration)
    # Step names and corresponding durations for each repetition
    step_names = ['Cobra Move', 'MCS', 'Link file']
    additional_step = 'FPS ops'
    step_names_with_additional = [additional_step] + step_names

    num_repetitions = duration.shape[0]
    x = np.arange(len(step_names))+1  # x-coordinates for the bars
    bar_width = 0.8 / num_repetitions  # Width of each bar

    plt.figure(figsize=(8, 6))
    plt.bar(0, fpsOps, width=bar_width, label=f'FPS init')

    # Create a bar plot for each repeat
    if get_total_iteration(visit)+1 == get_input_iteration(visit):
        for i in range(num_repetitions):
            plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Iteration {i+1}')
    else:
        for i in range(num_repetitions):
            if i == 0:
                plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Home Cobra')
            else:
                plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Iteration {i}')

    totalTime = (fpsDone[-1] - fpsStart[0]).total_seconds()    

    plt.xlabel('Steps')
    plt.ylabel('Time spent (seconds)')
    plt.title(f'FPS Time spent in each step (visit={visit}, t={totalTime}s)')
    plt.xticks([0,1,2,3], step_names_with_additional)
    plt.legend()
    plt.ylim(0,30)
    plt.tight_layout()  # Adjusts spacing between plot elements
    
    
    if figName is not None:
        plt.savefig(figName)
    else:
        plt.show()

def parse_mcs_log(pfsVisit, debug=False):
    
    directory_path = '/data/logs/actors/mcs/'
    search_string = f'frameId={pfsVisit}'

    mcsLogFile = find_log_file_with_string(directory_path, search_string)
    if debug:
        print(mcsLogFile)
    maxSubframe = get_subframe_maxium(pfsVisit, mcsLogFile=mcsLogFile)
    
    # Read the log file and extract the time stamps
    with open(mcsLogFile, 'r') as file:
        log_lines = file.readlines()
    
    start_analysis = False
    last_frame = False
    
    expStart=[]
    readDone = []
    saveDone = []
    expDone = []
    
    fidStart = []
    fidDone = []
    cenTime = []
    
    outFF = []
    allFF = []
    appTrans = []
    # Parse the time stamps and calculate the time spent
    for line in log_lines:
        if not start_analysis:
            if f'frameId={pfsVisit}00' in line:
                if debug: 
                    print(f'start_analysys = {start_analysis}')
                    print(line)
                start_analysis = True
                expStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
        else:
            if 'expose object' in line:
                if debug: print(line)
                expStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            elif 'newpath' in line:
                if debug: print(line)
                readDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                match = re.search(r'PFSC(\d+)\.fits', line)
                number = int(match.group(1))
                if number == pfsVisit*100+maxSubframe:
                    last_frame = True
                if debug: 
                    print(number,pfsVisit*100+maxSubframe,last_frame)
            elif 'hdr done' in line:
                if debug: print(line)
                saveDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                
            elif 'Sending centroid data to database' in line:
                cenTime.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if debug: print(line)
            #elif 'Calcuating transofmtaion using FF' in line:
            #    outFF.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            #    if debug: print(line)
            #elif 'Re-calcuating transofmtaion using ALL FFs.' in line:
            #    allFF.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            #    if debug: print(line)
            #elif 'Apply transformation to MCS data points' in line:
            #    appTrans.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            #    if debug: print(line)
            elif 'Starting Fiber ID' in line:
                if debug: print(line)
                fidStart.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            elif 'Fiber ID finished' in line:
                if debug: print(line)
                fidDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
            elif 'exposureState=done' in line:
                if debug: print(line)
                expDone.append(datetime.datetime.strptime(f'{line.split()[0]} {line.split()[1]}', '%Y-%m-%d %H:%M:%S.%fZ'))
                if last_frame is True:
                    start_analysis = False
    #return expStart, readDone, saveDone, cenTime, outFF, allFF, appTrans, fidStart, fidDone, expDone
    return expStart, readDone, saveDone, cenTime, fidStart, expDone
    #return expStart[:-1], readDone[:-1], saveDone[:-1], cenTime[:-1], fidStart[:-1], expDone[:-1]

def visMCSTimeBar(visit, figName=None):
    #expStart, readDone, saveDone, cenTime, outFF, allFF, appTrans, fidStart, fidDone, expDone = parse_mcs_log(visit)
    expStart, readDone, saveDone, cenTime, fidStart, expDone = parse_mcs_log(visit)

    duration = []
    for i in range(len(readDone)):
        duration.append(np.array([
            (readDone[i] - expStart[i]).total_seconds(), 
            (saveDone[i] - readDone[i]).total_seconds(),  
            (cenTime[i] - saveDone[i]).total_seconds(),
            #(allFF[i] - outFF[i]).total_seconds(),
            #(appTrans[i] - allFF[i]).total_seconds(),
            (fidStart[i] - cenTime[i]).total_seconds(),
            (expDone[i] - fidStart[i]).total_seconds(),        
            #(expDone[i] - fidDone[i]).total_seconds(),        
        ]))

    duration = np.array(duration)
    
        # Step names and corresponding durations for each repetition
    step_names = ['MCS exposure', 'FITS process', 'Centroid Time', 'Transform','Fiber ID']

    num_repetitions = duration.shape[0]
    x = np.arange(len(step_names))  # x-coordinates for the bars
    bar_width = 0.8 / num_repetitions  # Width of each bar
    plt.figure(figsize=(8, 6))

    # Create a bar plot for each repetition
    if get_total_iteration(visit)+1 == get_input_iteration(visit):
        for i in range(num_repetitions):
            plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Iteration {i+1}')
    else:
        for i in range(num_repetitions):
            if i == 0:
                plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Home Cobra')
            else:
                plt.bar(x + i*bar_width, duration[i], width=bar_width, label=f'Iteration {i}')

    plt.xlabel('Steps')
    plt.ylabel('Time spent (seconds)')
    plt.title(f'MCS Time spent in each step (visit={visit})')
    plt.xticks(x, step_names)
    plt.legend()
    plt.ylim(0,15)
    plt.tight_layout()  # Adjusts spacing between plot elements
    
    
    if figName is not None:
        plt.savefig(figName)
    else:
        plt.show()
    
