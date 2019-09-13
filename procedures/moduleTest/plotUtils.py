import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from procedures.moduleTest import calculation
from ics.cobraCharmer.utils import butler

def plotOntimeSet(moduleName, ontimeRuns, motor, stepSize):
    npl = 57
    nrows = (npl + 3)//4
    f,pl = plt.subplots(nrows=nrows, ncols=4, figsize=(16, nrows*4), sharey=True)
    pl = pl.flatten()

    tangs = dict()
    tangrs = dict()
    speeds = dict()
    cal = calculation.Calculation(butler.mapPathForModule(moduleName, version='phiOntime'), [], None)

    for ot in ontimeRuns.keys():
        tangs[ot] = np.load(ontimeRuns[ot] / 'data' / f'{motor}AngFW.npy')
        tangrs[ot] = np.load(ontimeRuns[ot] / 'data' / f'{motor}AngRV.npy')
        speeds[ot] = cal.speed(tangs[ot], tangrs[ot], stepSize)
    for i in range(npl):
        for ot in sorted(ontimeRuns.keys()):
            setName = f'{ot}us' if f != 999 else 'opt'
            pl[i].plot(np.rad2deg(tangs[ot][i][0]), '+-', label=f'{setName}: {speeds[ot][0][i]:0.4f}')
            pl[i].legend()
        pl[i].set_title(f'{moduleName} {motor} {i+1}')

    return tangs, tangrs, speeds

def plotConvergenceRuns(runPaths, motor):
    if isinstance(runPaths, (str, pathlib.Path)):
        runPaths = [runPaths]
    nRuns = len(runPaths)
    if nRuns == 1:
        nrows = 2
        ncols = 1
    else:
        nrows = nRuns
        ncols = 2

    f, pl = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                         num=f'{motor}convergenceTest',
                         figsize=(12, 2*nrows))

    for run_i, runPath in enumerate(runPaths):
        dataPath = sorted(runPath.glob(f'output/{motor}convergence.npy'))
        dataPath = dataPath[0]
        data = np.load(dataPath)
        runName = runPath.stem

        p1, p2 = pl[run_i]

        cobras =  np.unique(data['cobra'])
        lastIteration = np.max(data['iteration'])

        p1.hlines(0, 0, lastIteration + 1, 'k', alpha=0.2)
        for c_i in cobras:
            if c_i == 0:
                continue
            c_w = np.where(data['cobra'] == c_i)
            p1.plot(np.rad2deg(data['left'][c_w]), '-+', alpha=0.5)
            p1.set_title(runName)
            p1.set_ylabel('degrees')
            p1.xaxis.set_major_locator(MaxNLocator(integer=True))

        duds = np.where((data['iteration'] == lastIteration) & (data['done'] == False))

        if lastIteration >= 7 and len(duds[0]) > 0:
            skip = 1
            dudCobras = np.unique(data['cobra'][duds])
            x = np.arange(skip, lastIteration+1)
            p2.hlines(0, x[0], x[-1], 'k', alpha=0.2)

            for c_i in dudCobras:
                c_w = np.where(data['cobra'] == c_i)
                p2.plot(x, np.rad2deg(data['left'][c_w])[skip:], '-+', alpha=0.5, label=f'{c_i}')
                p2.set_title(f'{runName} failures')
                p2.set_ylabel('degrees')
                p2.xaxis.set_major_locator(MaxNLocator(integer=True))
                p2.legend()
        else:
            skip = 2 if lastIteration > 4 else 1
            x = np.arange(skip, lastIteration+1)
            p2.hlines(0, x[0], x[-1], 'k', alpha=0.2)
            for c_i in cobras:
                if c_i == 0:
                    continue
                c_w = np.where(data['cobra'] == c_i)
                p2.plot(x, np.rad2deg(data['left'][c_w][skip:]), '-+', alpha=0.5)
                p2.set_ylim(-5,5)
                p2.set_title(f'{runName} end moves')
                p2.set_ylabel('degrees')
                p2.xaxis.set_major_locator(MaxNLocator(integer=True))

        f.tight_layout()
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
