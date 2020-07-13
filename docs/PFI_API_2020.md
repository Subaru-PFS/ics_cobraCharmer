Main Goals
==========

- switch to unified PFI geometry and global ids, based on `pfs.utils.fiberIds` and `pfs.utils.coordinates`. In particular, get rid of all per-module logic and geometry, and use PFI mm and angles with respect to the PFI grid.
- use unified persistence mechanism based on `pfs.utils.butler.Butler`.
- convert existing Chih-Yi and Chi-Hung code to use per-cobra implementation. The algorithms of how to acquire motormaps or to run 1d or 2d convergence should not really be affected, but the internals of the implementations in `cobraCoach.py` and `engineer.py` will be.
- Decide on the API between `fpsActor` and the `PFI` class below. I'd like to say that loops go into FPS.
- Decide on `opDb` vs. `ics_instData` persistence.
- Find and understand all the places where we really want/need to treat cobras components as arrays: for pix-to-mm coordinate transforms, etc. Make sure we do not confuse ourselves when changing views, and make sure we do not destroy performance (a 400us 2400-cobra operation will take a second.)

Issues which are not covered
----------------------------

All these _matter_ and are related, but will be handled elsewhere.

- refactoring between `fpsActor`, `iicActor`, `mcsActor`, and the current upper-layer `cobraCharmer` routines (acquire motormaps, run convergence).
- the top-level FPS loops, except where we need to define the API.
- Details or usage of `pfs.utils.coordinates` transforms (pix-to-mm, etc.)
- `opDb` usage. Existing usage should not be broken.
- per-`run` logging details. Existing usage should not be broken.
- `pfs.utils.butler` details besides not breaking persistence.

class MotorMap
---------------

Holds knowledge of how best to drive a single motor in a single direction, in the sense of how many steps to request at what ontime.  Does know about sticky/fast/unreachable regions. Does know conditions when acquiring map (esp. motor frequencies). Does not know about geometry, neighbors, or collisions.

Holds several variable length arrays, all of the same length:
  - `angle`: the angles at the edges of the array slots.
  - `steps`: the number of steps to traverse the slot.
  - `ontime`: the ontimes to drive the motor at for that slot.
  - `flags`: TBD. `STICKY|CANNOT_REACH`, maybe, etc. This is for `MotorMap`-only flags, and **not** any which require knowledge about geometry or adjacent cobras. 
   
 It is up to the `Cobra`, etc. to select and maintain the `MotorMaps`.

Core API:
- `map.calculateMove(fromAngle, toAngle`): return a list of motion segments, each specifying `(int steps, float ontime)`. To start with it will return one segment calculated as in the existing Chih-Yi code. Note that the segments *could* be moved consecutively without taking images; that is up to the caller.
- `map.updateMap(steps, ontime)`: update the two internal vectors. Either can be None and left alone; ontime can be a single float. 
- `map.dump(path)`: persist map.
- `map = MotorMap.load(path)`: load a persisted map.

Persistence:
- Not sure that `opDb` tables are needed. Are they?
- Path comes from `pfs.utils.butler`, but is `f{motor}-f{direction}-f{mapName}.yaml` file, saved in per-cobra directory.

Big questions:
- Can `calculateMove()` always be self-contained: can we push all geometry and collision logic into the containing `Cobra`? I think so, but we should talk.
- Should scaling, etc. be kept in `MotorMap` or `Cobra`?

class PFI:
----------

*Note:* There already is a `pfi.py`, but it contains the fundamental path routines,
FPGA control, and cobra accessors. Pull the FPGA control out, move
path routines to `Cobra` and `MotorMap` classes, and cleanup the rest.

Holds collection of all (connected) Cobras, plus the fiducials. 
Provides all major top-level single-step functionality (driving a convergence step, motormap acquisition, simple homing, etc). Basically the main entrypoints in the existing modules, but operating on sets of `Cobras` instead of on individual modules. 

Most operations involve interrogating sets of `Cobras` and calling for `FPGA` moves.

Q: Can we really limit `PFI` operations to theta and phi except when matching to spots and converting targets coordinates from PFI mm? Want to say yes.
Q: Do we really want to limit `PFI` to single-step operations? Want to say yes.

Core API:

 - `matchCobrasToSpots(spots, cobras=None) -> cobras`: match measured spots to expected `Cobra` positions. This depends on the `Cobra.getExpectedPosition()` routine.
 - `moveToThetaPhi(theta, phi, cobras=None)`: move the given `Cobra`s to the given theta and phi angles w.r.t. the PFI grid. Single-step move: loops are handled externally.
 - `selectCobrasByXXX()`: return set of cobras for a single module, field, etc. By default we operate on all cobras on PFI, but we can also operate on subsets.
 - **etc**

class Cobra:
------------

- includes all cobra geometry, serial numbers, module assignment (CIT and PFI), status flags, etc.
- dynamically manages named forward and revese `MotorMaps` for each motor.
- probably knows 
- knows about neighbors (cobras, fiducials, inconvenient bolt-heads)
- tracks current position, dynamic ontimes, commanded moves. Arranges for `opDb` and logging, but updates probably aggregated by `PFI`.
- does **not** command moves itself, but has a method which returns what to ask for.

Core API:
- `getExpectedPosition(thetaAngle, phiAngle) -> x,y`: Return best-guess for position *after* move; this supports the global spot-to-fiber solver. Should we be returning annular ellipse instead of a point?
- `thetaPhiToXy(thetaAngle, phiAngle) -> x,y`: convert pfi MM to theta,phi
- `xyToThetaPhi(x, y) -> theta,phi`: convert theta, pfi to pfi MM
- `setMeasuredPosition(x, y)`: declare that we are at pfi X,Y. **Must** be called after every move+exposure.
- `calcMovesToThetaPhi(thetaAngle, phiAngle) -> thetaSegment, phiSegments`: query appropriate `MotorMaps` for path segments. We (`Cobra`) have to handle neighbors, etc.

Persistence:
- Some `odDb` tables exist.
- Path comes from `pfs.utils.butler.Butler`, but will boil down to `geometry.yaml` in `$ICS_INSTDATA_DIR/data/pfi/cobras/$cobraId`.

class FPGA:
-----------

Wrap the core FPGA commands without **any** other logic, and get rid
the existing cluster of stray modules (cmd.py, func.py, etc). Have it
use the Cobra class for any book-keeping (last move and position,
effective ontime, etc).

Core API:
- `moveCobraSegments(cobras, thetaMoves, phiMoves)`: command the given cobra moves. Note that each cobra is sent a _list_ of (ontime, steps) segments.
- the existing engineering routines.

fpsActor routines
-----------------

The top-level logic in the `cobraCharmer/procedures/moduleTest/{cobraCoach.py`, etc. modules needs to be copied to `fpsActor` and converted to use the above `PFI` architecture. We will for now _leave_ the existing routines in place for the existing CIT and ASRD bench operations.

Most routines will need actor commands (takeThetaMotorMap,
calibrateFrequencies, etc)

Convert existing convergence routines to use the `mcsActor` camera and
consume the `mcs_data` spot table.

Tickets
-------

In some crude time order, a growing list of tickets to write up:

 - implement a `pfs_utils.coordinates` compatible transform for the ASRD MCS camera. [ Fiducials being measured now. Richard gets map, gives to Chi-Hung ] [Done?]
 - add cobraId -> module,etc mapping functions to `pfs_utils.fiberIds` (for FPGA calls) [CPL]
 - update opDb tables for PFI control [CH, then hand to Yabe]
 - make above PFI, MotorMap, FPGA, and Cobra classes [CPL]
 - create `fpsActor` routines, based on the existing
   `cobraCharmer.procedures.moduleTest` code, but using new classes and
    leaving behind dead code. [CH, CY, CPL?]
 - make xyToThetaPhi(x,y) [now `pfi.positionsToAngles()`] non-exact, robust. [CY]
 - make scripts to generate `pfs_config` tables for engineering
   (convergence tests, various interesting positions, etc) [later]
 - add routine to provide per-cobra and per-module convergence metrics
   (green light S/N for now). [CH]

Strays
------

- update `pfs.utils.butler` and maps as necessary. 
- continue performance checking on `opDb` due to restructuring. Prepare to add indices or use SQL ARRAYs if we must.

2020-07-12 CPL
