import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plotRanges(model, moduleName):
    """ Plot the motor ranges and anomalies. """

    phiRange = 180,200
    thetaRange = 360,400
    phiPlotRange = 150,200
    thetaPlotRange = 350,400

    dPhi = model.phiOut - model.phiIn
    dPhi = np.rad2deg(dPhi)

    # This is nuts:
    dTheta = model.tht1 - model.tht0
    dTheta = np.rad2deg(dTheta)
    dTheta[dTheta < 0] += 360
    dTheta[dTheta > 400] -= 360
    dTheta[(dTheta < 40) & (dTheta > 0)] += 360

    OR = np.logical_or
    AND = np.logical_and

    brokenFibers = (model.status & model.COBRA_INVISIBLE_MASK) != 0
    dudPhis = AND(~brokenFibers, OR(dPhi < phiRange[0], dPhi > phiRange[1]))
    dudThetas = AND(~brokenFibers, OR(dTheta < thetaRange[0], dTheta > thetaRange[1]))
    haveDudMotors = np.any(dudPhis | dudThetas)

    nrows = (1 + haveDudMotors)
    f, pl = plt.subplots(ncols=2, nrows=nrows,
                         figsize=(8, 4*nrows), squeeze=False)
    cobraRange = np.arange(len(dPhi)) + 1

    p1,p2 = pl[0]
    p1.hlines(180, 0, len(dPhi), color='k', alpha=0.4)
    p1.plot(cobraRange, dPhi, 'o', alpha=0.5)
    p1.set_ylim(phiPlotRange)
    p1.set_ylabel('degrees')
    p1.set_title(f'{moduleName} phi')
    p1.grid(b=True, axis='x', alpha=0.5)

    p2.hlines(360, 0, len(dTheta), color='k', alpha=0.4)
    p2.plot(cobraRange, dTheta, 'o', alpha=0.5)
    p2.set_ylim(thetaPlotRange)
    p2.set_ylabel('degrees')
    p2.set_title(f'{moduleName} theta')
    p2.grid(b=True, axis='x', alpha=0.5)

    if haveDudMotors:
        dp1, dp2 = pl[1]

        if np.any(dudPhis):
            phiIdx = np.where(dudPhis)[0]
            print(f"{moduleName} phiduds: {phiIdx}")
            dp1.hlines(180, 0, len(dPhi), color='k', alpha=0.4)
            dp1.plot(phiIdx + 1, dPhi[phiIdx], 'o',
                     color='red', alpha=0.8)
            p1.vlines(phiIdx+1, *phiPlotRange, label=f'bad range {phiIdx+1}',
                      color='red', alpha=0.5, linestyle='dotted')
            dp1.set_ylabel('degrees')
            dp1.set_title(f'{moduleName} bad phi range')
            dp1.grid(b=True, axis='x', alpha=0.5)

            dp1.legend()
        else:
            dp1.set_visible(False)

        if np.any(dudThetas):
            thetaIdx = np.where(dudThetas)[0]
            print(f"{moduleName} thetaduds: {thetaIdx}")
            dp2.hlines(360, 0, len(dTheta), color='k', alpha=0.4)
            dp2.plot(thetaIdx + 1, dTheta[thetaIdx], 'o',
                     color='red', alpha=0.8)
            p2.vlines(thetaIdx+1, *thetaPlotRange, label=f'bad range {thetaIdx+1}',
                      color='red', alpha=0.5, linestyle='dotted')
            dp2.set_ylabel('degrees')
            dp2.set_title(f'{moduleName} bad theta range')
            dp2.grid(b=True, axis='x', alpha=0.5)

            dp2.legend()
        else:
            dp2.set_visible(False)

    if np.any(brokenFibers):
        brokenFiberIdx = np.where(brokenFibers)[0]
        p1.vlines(brokenFiberIdx+1, *phiPlotRange, label=f'broken fibers {brokenFiberIdx+1}',
                  color='magenta', alpha=0.4, linestyle='dotted')
        p2.vlines(brokenFiberIdx+1, *thetaPlotRange, label=f'broken fibers {brokenFiberIdx+1}',
                  color='magenta', alpha=0.4, linestyle='dotted')

    p1.legend()
    p2.legend()

    f.tight_layout()
    return f

def plotOntimeSet(moduleName, ontimeRuns, motor, stepSize):
    npl = 57
    nrows = (npl + 3)//4
    f,pl = plt.subplots(nrows=nrows, ncols=4, figsize=(16, nrows*4), sharey=True)
    pl = pl.flatten()

    f.suptitle(f'{moduleName} {motor} ontimes')
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

def plotConvergenceRuns(runPaths, motor, endWidth=2.0, convergence=np.rad2deg(0.005), moduleName=""):
    if isinstance(runPaths, (str, pathlib.Path)):
        runPaths = [runPaths]
    runPaths =  sorted(runPaths)
    nRuns = len(runPaths)
    nrows = nRuns
    ncols = 3

    title = f'{moduleName} {motor} to {convergence:0.2f} deg, from run {runPaths[0].stem}'
    f, pl = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                         squeeze=False,
                         num=f'{motor}convergenceTest',
                         figsize=(12, 2*nrows))
    f.suptitle(title, fontsize=15)

    for run_i, runPath in enumerate(runPaths):
        dataPath = sorted(runPath.glob(f'output/{motor}Convergence.npy'))
        dataPath = dataPath[0]
        data = np.load(dataPath)
        runName = runPath.stem

        p1, p2, p3 = pl[run_i]

        cobras =  np.unique(data['cobra'])
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

            pp = p1.plot(x, np.rad2deg(data['left'][c_w][iterIdx]), '-+', alpha=0.5)
            p1.set_title(f'run {run_i+1} phi {np.rad2deg(data["position"][c_w[0]]):0.2f} to '
                         f'{np.rad2deg(data["target"][c_w[0]]):0.2f}')
            p1.set_ylabel('degrees')
            p1.set_xlim(-0.5,8.5)
            p1.xaxis.set_major_locator(MaxNLocator(integer=True))

            color = pp[0].get_color()
            p2.plot(x, np.rad2deg(data['left'][c_w][iterIdx]), '-+', color=color, alpha=0.5)
            p2.set_ylim(-endWidth, endWidth)
            p2.set_title('end moves')
            p2.xaxis.set_major_locator(MaxNLocator(integer=True))

            dudSkip = 1
            if isDud:
                stepsOff = data['steps'][c_w[-1]]
                p3.plot(x[dudSkip:], np.rad2deg(data['left'][c_w])[dudSkip:], '-+',
                        color=color, alpha=0.5, label=f'{c_i}')
                p3.set_title('failures')
                p3.xaxis.set_major_locator(MaxNLocator(integer=True))

        if haveDud:
            p3.hlines(-convergence, 0, 8, 'k', alpha=0.1)
            p3.hlines(convergence, 0, 8, 'k', alpha=0.1)
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
