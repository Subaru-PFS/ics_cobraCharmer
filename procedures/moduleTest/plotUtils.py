import logging
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ics.cobraCharmer.utils import butler

def plotOntimeSet(moduleName, ontimeRuns, motor, stepSize):
    npl = 57
    nrows = (npl + 3)//4
    f,pl = plt.subplots(nrows=nrows, ncols=4, figsize=(16, nrows*4), sharey=True)
    pl = pl.flatten()

    tangs = dict()
    tangrs = dict()

    if motor == 'phi':
        yrange=(-200,200)
    else:
        yrange=(-400,400)

    for ot in sorted(ontimeRuns.keys()):
        logging.info(f"{ot}: {ontimeRuns[ot]}")
        tangs[ot] = np.load(ontimeRuns[ot] / 'data' / f'{motor}AngFW.npy')
        tangrs[ot] = np.load(ontimeRuns[ot] / 'data' / f'{motor}AngRV.npy')

    for i in range(npl):
        for ot in sorted(ontimeRuns.keys()):
            kw = dict()
            if ot == 999:
                setName = 'opt'
                kw['color'] = 'k'
                kw['linewidth'] = 1.5
            else:
                setName = f'{ot}us'
                kw['marker'] = '+'
            fw = np.rad2deg(tangs[ot][i][0])
            rv = np.rad2deg(tangrs[ot][i][0] - max(tangrs[ot][i][0]))

            if isinstance(stepSize, dict):
                stepSize1 = stepSize[ot]
            else:
                stepSize1 = stepSize

            x = stepSize1 * np.arange(len(fw))
            ll = pl[i].plot(x, fw, ls='-', label=f'{setName}', **kw)
            if 'color' in kw:
                kw.pop('color')
            pl[i].plot(x, rv, ls='-', color=ll[0].get_color(), **kw)
            pl[i].legend()
        pl[i].hlines(0, x[0], x[-1], alpha=0.4)
        pl[i].set_ylim(*yrange)
        pl[i].set_ylabel('degrees')
        pl[i].set_title(f'{moduleName} {motor} {i+1}')

    return f

def plotConvergenceRuns(runPaths, motor, endWidth=2.0, convergence=np.rad2deg(0.005)):
    if isinstance(runPaths, (str, pathlib.Path)):
        runPaths = [runPaths]
    nRuns = len(runPaths)
    nrows = nRuns
    ncols = 3

    f, pl = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                         squeeze=False,
                         num=f'{motor}convergenceTest',
                         figsize=(12, 2*nrows))

    for run_i, runPath in enumerate(runPaths):
        dataPath = sorted(runPath.glob(f'output/{motor}Convergence.npy'))
        dataPath = dataPath[0]
        data = np.load(dataPath)
        runName = runPath.stem

        p1, p2, p3 = pl[run_i]

        cobras =  np.unique(data['cobra'])
        lastIteration = np.max(data['iteration'])
        haveDud = False

        p1.hlines(0, 0, 8, 'k', alpha=0.2)
        p2.hlines(0, 0, 8, 'k', linestyle=':', alpha=0.05)
        p2.hlines(-convergence, 0, 8, 'k', alpha=0.1)
        p2.hlines(convergence, 0, 8, 'k', alpha=0.1)
        for c_i in cobras:
            if c_i == 0:
                continue
            c_w = np.where(data['cobra'] == c_i)[0]
            done_w = np.where(data['done'][c_w])[0]
            isDud = len(done_w) == 0
            if isDud:
                haveDud = True
                maxIter = np.max(data['iteration'][c_w])
            else:
                maxIter =  np.min(done_w)

            iterIdx = np.arange(maxIter+1)
            x = iterIdx

            p1.plot(x, np.rad2deg(data['left'][c_w][iterIdx]), '-+', alpha=0.5)
            p1.set_title(runName)
            p1.set_ylabel('degrees')
            p1.set_xlim(-0.5,8.5)
            p1.xaxis.set_major_locator(MaxNLocator(integer=True))

            p2.plot(x, np.rad2deg(data['left'][c_w][iterIdx]), '-+', alpha=0.5)
            p2.set_ylim(-endWidth, endWidth)
            p2.set_title(f'{runName} end moves')
            p2.xaxis.set_major_locator(MaxNLocator(integer=True))

            dudSkip = 1
            if isDud:
                p3.plot(x[dudSkip:], np.rad2deg(data['left'][c_w])[dudSkip:], '-+', alpha=0.5, label=f'{c_i}')
                p3.set_title(f'{runName} failures')
                p3.xaxis.set_major_locator(MaxNLocator(integer=True))

        if haveDud:
            p3.hlines(0, 0, 9, 'k', alpha=0.2)
            p3.legend()

        # f.tight_layout()
    return f, data

def plotConvergenceSummary(runPaths):
    if isinstance(runPaths, (str, pathlib.Path)):
        runPaths = [runPaths]
    nRuns = len(runPaths)

    f, pl = plt.subplots(nrows=2,
                         num='convergenceSummary',
                         figsize=(12, 8))
    p1, p2 = pl

    cobras = set()
    lastIteration = 0
    for run_i, dataPath in enumerate(runPaths):
        data = np.load(dataPath)

        cobras1 =  np.unique(data['cobra'])
        cobras.update(cobras1.tolist())
        lastIteration1 = np.max(data['iteration'])
        lastIteration = max(lastIteration, lastIteration1)

    cobras = sorted(cobras)
    doneAt = np.zeros((max(cobras)+1, nRuns), dtype='i4')

    for run_i, dataPath in enumerate(runPaths):
        data = np.load(dataPath)

        for i in range(lastIteration, 0, -1):
            i_w = np.where((data['iteration'] == i) & (data['done'] == True))
            doneCobras = data['cobra'][i_w]
            for c_i in doneCobras:
                doneAt[doneCobras,run_i] = i

    _ = p1.hist(doneAt.T, histtype='barstacked', bins=np.arange(np.max(doneAt)+1),
                density=True, align='left')
    p1.set_ylabel(f'percent of {len(cobras)} cobras in {nRuns} runs')
    p1.set_xlabel('steps to convergence')

    return doneAt
