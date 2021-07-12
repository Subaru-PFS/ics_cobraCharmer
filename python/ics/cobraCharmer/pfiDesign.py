"""

Modified from ics.cobraOps.CobrasCalibrationProduct
The values here are not modified from the XML file, so there
is no pixel scaling or phi home hacking. The coordinate
system here should be in F3C.

"""
from importlib import reload
import logging

import numpy as np
import xml.etree.ElementTree as ElementTree
from copy import deepcopy

from .utils import butler
reload(butler)


class PFIDesign():
    """ Class describing a cobras calibration product, the "motor map"  """

    COBRA_OK_MASK           = 0x0001  # a synthetic summary bit: 1 for good, 0 for bad.
    COBRA_INVISIBLE_MASK    = 0x0002  # 1 if the fiber is not visible
    COBRA_BROKEN_THETA_MASK = 0x0004  # 1 if the phi motor do not work
    COBRA_BROKEN_PHI_MASK   = 0x0008  # 1 if the theta motor does not work

    COBRA_BROKEN_MOTOR_MASK = COBRA_BROKEN_THETA_MASK | COBRA_BROKEN_PHI_MASK

    def __init__(self, fileName=None):
        """
        Constructs a new cobras calibration product using the information
        contained in a single XML calibration file.

        Parameters
        ----------
        fileName: str
            a path to an XML calibration file. [Deprecated]

        Returns
        -------
        object
            The cobras calibration product.

        Notes
        -----

        Since we are loading a single module, assume that we are
        directly connected to an FPGA, and thus that the module ID
        must be set to 1. If you want to _keep_ the module IDs, use
        PFIDesign.loadPFI().
        """

        self.logger = logging.getLogger('pfiDesign')

        if fileName is None:
            return

        self.loadModelFiles([fileName])
        #self.moduleIds[:] = 1

    @classmethod
    def loadModule(cls, moduleName, version=None):
        """ """
        modulePath = butler.mapPathForModule(moduleName, version=version)

        return cls(modulePath)

    @classmethod
    def loadPfi(cls, version=None, moduleVersion=None):
        """ """
        moduleNames = butler.modulesForPfi(version=version)

        modulePaths = []
        for m in moduleNames:
            modulePaths.append(butler.mapPathForModule(m, version=moduleVersion))

        self = cls(None)
        self.loadModelFiles(modulePaths)

        return self

    def fixModuleIds(self):
        """
        During the testing phase, not all cobra modules are installed.
        We may want to test any modules. This one change the module IDs
        to (1 .. number of modules) in the same order.
        """
        mod = 0
        for i in range(self.nCobras):
            if i % 57 == 0:
                mod += 1
            self.moduleIds[i] = mod

        return mod

    def _loadCobrasFromModelFile(self, fileName):
        """Loads the per-cobra structures from the given model file

        Parameters
        ----------
        fileName: object
            The path to the XML calibration file.

        Returns
        -------
        arms : list of ARM_DATA_CONTAINERs
            The cobras calibration product.
        info : dictionary
            Some metadata which might be useful.
        """

        # Load the XML calibration file
        calibrationFileRootElement = ElementTree.parse(fileName).getroot()

        # Get all the data container elements
        dataContainers = calibrationFileRootElement.findall("ARM_DATA_CONTAINER")

        # Grab "metadata"
        info = dict()
        try:
            info['name'] = calibrationFileRootElement.find('ARM_DATA_NAME').text
        except AttributeError:
            info['name'] = fileName.stem
        try:
            info['site'] = calibrationFileRootElement.find('ARM_DATA_SITE').text
        except AttributeError:
            info['site'] = 'unknown'

        return dataContainers, info

    def loadModelFiles(self, fileList):
        """Constructs a new cobras calibration product using the information
        contained in a list of XML calibration file.

        Parameters
        ----------
        fileNames: list
            The full paths to the XML calibration files.

        Returns
        -------
        object
            The cobras calibration product.

        """

        dataContainers = []
        self.modelInfo = {}
        for f in fileList:
            fileArms, fileInfo = self._loadCobrasFromModelFile(f)
            dataContainers.extend(fileArms)
            self.modelInfo[f] = fileInfo

        self.origin_dataContainers = dataContainers
        self.dataContainers = deepcopy(dataContainers)

        # The number of cobras is equal to the number of data containers
        self.nCobras = len(dataContainers)

        # Create some of the calibration data arrays
        self.moduleIds = np.empty(self.nCobras, dtype="u2")
        self.positionerIds = np.empty(self.nCobras, dtype="u2")
        self.serialIds = np.empty(self.nCobras, dtype="u2")
        self.centers = np.empty(self.nCobras, dtype="complex")
        self.status = np.empty(self.nCobras, dtype="u2")
        self.tht0 = np.empty(self.nCobras)
        self.tht1 = np.empty(self.nCobras)
        self.phiIn = np.empty(self.nCobras)
        self.phiOut = np.empty(self.nCobras)
        self.L1 = np.empty(self.nCobras)
        self.L2 = np.empty(self.nCobras)

        self.motorFreq1 = np.empty(self.nCobras)
        self.motorFreq2 = np.empty(self.nCobras)

        self.motorOntimeFwd1 = np.empty(self.nCobras)
        self.motorOntimeFwd2 = np.empty(self.nCobras)
        self.motorOntimeRev1 = np.empty(self.nCobras)
        self.motorOntimeRev2 = np.empty(self.nCobras)

        self.motorOntimeSlowFwd1 = np.empty(self.nCobras)
        self.motorOntimeSlowFwd2 = np.empty(self.nCobras)
        self.motorOntimeSlowRev1 = np.empty(self.nCobras)
        self.motorOntimeSlowRev2 = np.empty(self.nCobras)

        # Check if the data containers have information about the motor maps
        slowCalTable = dataContainers[0].find("SLOW_CALIBRATION_TABLE")

        if slowCalTable is not None and slowCalTable.find("Joint1_fwd_stepsizes") is not None:
            # The number of motor map steps is saved in the first element of
            # the arrays
            self.motorMapSteps = int(slowCalTable.find("Joint1_fwd_stepsizes").text.split(",")[0])

            # Create the cobra motor map arrays
            self.angularSteps = np.empty(self.nCobras)
            self.S1Pm = np.empty((self.nCobras, self.motorMapSteps))
            self.S1Nm = np.empty((self.nCobras, self.motorMapSteps))
            self.S2Pm = np.empty((self.nCobras, self.motorMapSteps))
            self.S2Nm = np.empty((self.nCobras, self.motorMapSteps))
            self.F1Pm = np.empty((self.nCobras, self.motorMapSteps))
            self.F1Nm = np.empty((self.nCobras, self.motorMapSteps))
            self.F2Pm = np.empty((self.nCobras, self.motorMapSteps))
            self.F2Nm = np.empty((self.nCobras, self.motorMapSteps))

        # Fill the cobras calibration arrays
        for i in range(self.nCobras):
            # Save some of the data header information
            header = dataContainers[i].find("DATA_HEADER")
            self.moduleIds[i] = int(header.find("Module_Id").text)
            self.positionerIds[i] = int(header.find("Positioner_Id").text)
            self.serialIds[i] = int(header.find("Serial_Number").text, base=10)
            self.status[i] = int(header.find("Status").text, base=10)

            # Check for conflicts:
            for check_i in range(i):
                if (self.moduleIds[i] == self.moduleIds[check_i] and
                        self.positionerIds[i] == self.positionerIds[check_i]):

                    raise KeyError(f"duplicate cobra id: module={self.moduleIds[i]} "
                                   f"positioner={self.positionerIds[i]}")

                if self.serialIds[i] == self.serialIds[check_i]:
                    raise KeyError(
                        f"duplicate cobra with serial={self.serialIds[i]} in SC{self.moduleIds[i]}PID{self.positionerIds[i]} and SC{self.moduleIds[check_i]}PID{self.positionerIds[check_i]}")

            # Save some of the kinematics information
            kinematics = dataContainers[i].find("KINEMATICS")
            self.centers[i] = float(kinematics.find("Global_base_pos_x").text) + \
                float(kinematics.find("Global_base_pos_y").text) * 1j
            self.tht0[i] = np.deg2rad(float(kinematics.find("CCW_Global_base_ori_z").text))
            self.tht1[i] = np.deg2rad(float(kinematics.find("CW_Global_base_ori_z").text))
            self.phiIn[i] = np.deg2rad(float(kinematics.find("Joint2_CCW_limit_angle").text)) - np.pi
            self.phiOut[i] = np.deg2rad(float(kinematics.find("Joint2_CW_limit_angle").text)) - np.pi
            self.L1[i] = float(kinematics.find("Link1_Link_Length").text)
            self.L2[i] = float(kinematics.find("Link2_Link_Length").text)

            # Save the motor calibration information
            if hasattr(self, "motorMapSteps"):
                # Get the angular step used in the measurements
                slowCalTable = dataContainers[i].find("SLOW_CALIBRATION_TABLE")
                fastCalTable = dataContainers[i].find("FAST_CALIBRATION_TABLE")
                angularPositions = slowCalTable.find("Joint1_fwd_regions").text.split(",")[2:-1]
                angularStep = float(angularPositions[1]) - float(angularPositions[0])

                # Get the motor properties: frequencies and stepsizes
                self.motorFreq1[i] = float(header.find('Motor1_Run_Frequency').text)
                self.motorFreq2[i] = float(header.find('Motor2_Run_Frequency').text)

                self.motorOntimeFwd1[i] = float(kinematics.find('Link1_fwd_Duration').text)
                self.motorOntimeRev1[i] = float(kinematics.find('Link1_rev_Duration').text)
                self.motorOntimeFwd2[i] = float(kinematics.find('Link2_fwd_Duration').text)
                self.motorOntimeRev2[i] = float(kinematics.find('Link2_rev_Duration').text)

                self.motorOntimeSlowFwd1[i] = float(kinematics.find('Link1_fwd_Duration_Slow').text)
                self.motorOntimeSlowRev1[i] = float(kinematics.find('Link1_rev_Duration_Slow').text)
                self.motorOntimeSlowFwd2[i] = float(kinematics.find('Link2_fwd_Duration_Slow').text)
                self.motorOntimeSlowRev2[i] = float(kinematics.find('Link2_rev_Duration_Slow').text)

                # Get the cobra motors speeds in degrees per step
                slowJoint1Fwd = slowCalTable.find("Joint1_fwd_stepsizes").text.split(",")[2:-1]
                slowJoint1Rev = slowCalTable.find("Joint1_rev_stepsizes").text.split(",")[2:-1]
                slowJoint2Fwd = slowCalTable.find("Joint2_fwd_stepsizes").text.split(",")[2:-1]
                slowJoint2Rev = slowCalTable.find("Joint2_rev_stepsizes").text.split(",")[2:-1]
                fastJoint1Fwd = fastCalTable.find("Joint1_fwd_stepsizes").text.split(",")[2:-1]
                fastJoint1Rev = fastCalTable.find("Joint1_rev_stepsizes").text.split(",")[2:-1]
                fastJoint2Fwd = fastCalTable.find("Joint2_fwd_stepsizes").text.split(",")[2:-1]
                fastJoint2Rev = fastCalTable.find("Joint2_rev_stepsizes").text.split(",")[2:-1]

                # Calculate the motor steps required to move that angular step
                self.S1Pm[i] = angularStep / np.array(list(map(float, slowJoint1Fwd)))
                self.S1Nm[i] = angularStep / np.array(list(map(float, slowJoint1Rev)))
                self.S2Pm[i] = angularStep / np.array(list(map(float, slowJoint2Fwd)))
                self.S2Nm[i] = angularStep / np.array(list(map(float, slowJoint2Rev)))
                self.F1Pm[i] = angularStep / np.array(list(map(float, fastJoint1Fwd)))
                self.F1Nm[i] = angularStep / np.array(list(map(float, fastJoint1Rev)))
                self.F2Pm[i] = angularStep / np.array(list(map(float, fastJoint2Fwd)))
                self.F2Nm[i] = angularStep / np.array(list(map(float, fastJoint2Rev)))

                # Save the angular step in radians
                self.angularSteps[i] = np.deg2rad(angularStep)

        if hasattr(self, "motorMapSteps"):
            # Set the theta and phi offset arrays
            self.thtOffsets = np.arange(self.S1Pm.shape[1] + 1) * self.angularSteps[:, np.newaxis]
            self.phiOffsets = np.arange(self.S2Pm.shape[1] + 1) * self.angularSteps[:, np.newaxis]

            # Calculate the cumulative step sums
            zeros = np.zeros((self.nCobras, 1))
            self.posThtSlowSteps = np.hstack((zeros, np.cumsum(self.S1Pm, axis=1)))
            self.negThtSlowSteps = np.hstack((zeros, np.cumsum(self.S1Nm, axis=1)))
            self.posPhiSlowSteps = np.hstack((zeros, np.cumsum(self.S2Pm, axis=1)))
            self.negPhiSlowSteps = np.hstack((zeros, np.cumsum(self.S2Nm, axis=1)))
            self.posThtSteps = np.hstack((zeros, np.cumsum(self.F1Pm, axis=1)))
            self.negThtSteps = np.hstack((zeros, np.cumsum(self.F1Nm, axis=1)))
            self.posPhiSteps = np.hstack((zeros, np.cumsum(self.F2Pm, axis=1)))
            self.negPhiSteps = np.hstack((zeros, np.cumsum(self.F2Nm, axis=1)))

    def findAllCobras(self):
        return range(self.nCobras)

    def findCobraByCobraIndex(self, cobraIdx, readable=False):
        """ Find cobra at a given module and positioner."""

        cobraNum = np.array(cobraIdx) + 1
        cobraList = []
        for i in range(len(cobraNum)):
            ModuleID = int(cobraNum[i]/57)+1
            positionerId = cobraNum[i] % 57

            if positionerId == 0:
                ModuleID -= 1
                positionerId = 57
            if readable is True:
                cobraList.append(f'SC{ModuleID}/PID{positionerId:02d}')
            else:
                cobraList.append([ModuleID, positionerId])

        cobraList = np.array(cobraList)

        return cobraList

    def findCobraByModuleAndPositioner(self, moduleId, positionerId):
        """ Find cobra at a given module and positioner.

        Args
        ----
        moduleId : int
          The 1..42 number of a PFI module
        positionerId : int
          The 1..57 number of a module cobra

        Returns
        -------
        id : int
          The index into our data for the given cobra
        """

        return np.where((self.moduleIds == moduleId) & (self.positionerIds == positionerId))[0][0]

    def findCobraBySerialNumber(self, serialNumber):
        """ Find cobra with the given serial number.

        Args
        ----
        serialNumber : str
           The serial number of a cobra.

        Returns
        -------
        id : int
          The index into our data for the given cobra
        """

        return np.where(self.serialIds == serialNumber)[0][0]

    @staticmethod
    def getRealModuleId(moduleId):
        """ Get the canonical module id for a module id or name

        Args
        ---
        moduleId : int or str
          A module number (1..42), "SCxx", "Spare[12]"

        Spare[12] are assigned moduleIds 43 and 44.
        """

        if isinstance(moduleId, str):
            modName = moduleId.upper()
            if modName.startswith('SC'):
                moduleId = int(modName[2:], base=10)
            elif modName.startswith('SPARE'):
                spareId = int(modName[5:], base=10)
                moduleId = spareId + 42
            else:
                raise ValueError(f'invalid module name: {moduleId}')

        if moduleId < 1 or moduleId > 44:
            raise ValueError(f'module id out of range (1..44): {moduleId}')

        return moduleId

    def findCobrasForModule(self, moduleId):
        """ Find all cobras for a given module.

        Args
        ----
        moduleId : int
          The 1..42 number of a PFI module

        Returns
        -------
        ids : array
          The indices into our data for all cobras in the module
        """
        moduleId = self.getRealModuleId(moduleId)
        return np.where(self.moduleIds == moduleId)[0]

    def setCobraStatus(self, cobraId, moduleId=1, brokenTheta=False, brokenPhi=False, invisible=False):
        """ Set the operational status of a cobra.
        """
        cobraIdx = self.findCobraByModuleAndPositioner(moduleId, cobraId)

        self.status[cobraIdx] = self.COBRA_OK_MASK
        if invisible:
            self.status[cobraIdx] |= self.COBRA_INVISIBLE_MASK
            self.status[cobraIdx] &= ~self.COBRA_OK_MASK
        if brokenTheta:
            self.status[cobraIdx] |= self.COBRA_BROKEN_THETA_MASK
            self.status[cobraIdx] &= ~self.COBRA_OK_MASK
        if brokenPhi:
            self.status[cobraIdx] |= self.COBRA_BROKEN_PHI_MASK
            self.status[cobraIdx] &= ~self.COBRA_OK_MASK

        # Arrange for the new value to be persisted.
        header = self.dataContainers[cobraIdx].find("DATA_HEADER")
        header.find("Status").text = str(self.status[cobraIdx])

    def cobraStatus(self, cobraId, moduleId=1):
        cobraIdx = self.findCobraByModuleAndPositioner(moduleId, cobraId)
        return self.status[cobraIdx]

    def cobraIsGood(self, cobraId, moduleId=1):
        """ Return the cobra's status field. """

        status = self.cobraStatus(cobraId, moduleId=moduleId)
        return status == self.COBRA_OK_MASK

    def cobraIsBad(self, cobraId, moduleId=1):
        """ Return True if we believe cobra can/should NOT be used. """

        return not self.cobraIsGood(cobraId, moduleId=moduleId)

    def motorIsBroken(self, cobraId, moduleId=1):
        """ Return True if we believe cobra can/should NOT be moved. """

        status = self.cobraStatus(cobraId, moduleId=moduleId)
        return status & self.COBRA_BROKEN_MOTOR_MASK != 0

    def fiberIsBroken(self, cobraId, moduleId=1):
        """ Return True if we believe fiber cannot be seen. """

        status = self.cobraStatus(cobraId, moduleId=moduleId)
        return status & self.COBRA_INVISIBLE_MASK != 0

    def setModuleId(self, moduleId, forModule=None, setOurModuleIds=False):
        """ Update moduleIds

        Args
        ----
        moduleId: int or str
            The moduleId to set ourselves to.
        forModule: int
            If set, the module's IDs to (re)set.
        setOurModuleIds : bool
            Set both the container's module id AND our id (the ID on the FPGA bus).
        """

        moduleId = self.getRealModuleId(moduleId)
        if forModule is not None:
            idx = self.findCobrasForModule(forModule)
        else:
            idx = range(self.nCobras)

        # A length test is probably sufficient
        if len(idx) != 57:
            raise RuntimeError("Will not set moduleId for anything other than all cobras in a single module")
        for i in idx:
            header = self.dataContainers[i].find("DATA_HEADER")
            header.find("Module_Id").text = str(moduleId)

            if setOurModuleIds:
                self.moduleIds[i] = moduleId

        return self.moduleIds[0]

    def copyMotorMap(self, otherModel, motorIndex, doThetaFwd=False, doThetaRev=False,
                     doPhiFwd=False, doPhiRev=False, doFast=False):
        """ Copy maps for a given cobra from another model. """

        i = motorIndex

        if doFast:
            calTable = self.dataContainers[i].find("FAST_CALIBRATION_TABLE")
            otherTable = otherModel.dataContainers[i].find("FAST_CALIBRATION_TABLE")
        else:
            calTable = self.dataContainers[i].find("SLOW_CALIBRATION_TABLE")
            otherTable = otherModel.dataContainers[i].find("FAST_CALIBRATION_TABLE")

        if doThetaFwd:
            calTable.find("Joint1_fwd_stepsizes").text = otherTable.find("Joint1_fwd_stepsizes").text
        if doThetaRev:
            calTable.find("Joint1_rev_stepsizes").text = otherTable.find("Joint1_rev_stepsizes").text
        if doPhiFwd:
            calTable.find("Joint2_fwd_stepsizes").text = otherTable.find("Joint2_fwd_stepsizes").text
        if doPhiRev:
            calTable.find("Joint2_rev_stepsizes").text = otherTable.find("Joint2_rev_stepsizes").text

    def updateMotorMaps(self, thtFwd=None, thtRev=None, phiFwd=None, phiRev=None, useSlowMaps=True):
        """Update cobra motor maps

        Parameters
        ----------
        thtFwd: object
            A numpy array with forward step sizes for the theta motors.
        thtRev: object
            A numpy array with reverse step sizes for the theta motors.
        phiFwd: object
            A numpy array with forward step sizes for the phi motors.
        phiRev: object
            A numpy array with reverse step sizes for the phi motors.
        useSlowMaps: bool, optional
            If True (False), the slow (fast) motor maps will be used to
            update. Default is True.

        Note: Here we assume that all maps are using the same angular step size.
        Note: Parameters are in Radians

        """

        if thtFwd is not None and thtFwd.shape != (self.nCobras, self.motorMapSteps):
            raise RuntimeError("number of cobra theta forward motor maps must match number of cobras")
        if thtRev is not None and thtRev.shape != (self.nCobras, self.motorMapSteps):
            raise RuntimeError("number of cobra theta reverse motor maps must match number of cobras")
        if phiFwd is not None and phiFwd.shape != (self.nCobras, self.motorMapSteps):
            raise RuntimeError("number of cobra phi forward motor maps must match number of cobras")
        if phiRev is not None and phiRev.shape != (self.nCobras, self.motorMapSteps):
            raise RuntimeError("number of cobra phi reverse motor maps must match number of cobras")

        def f2s(x):
            return f'{x:.6f}'

        for i in range(self.nCobras):
            if useSlowMaps:
                calTable = self.dataContainers[i].find("SLOW_CALIBRATION_TABLE")
            else:
                calTable = self.dataContainers[i].find("FAST_CALIBRATION_TABLE")

            if thtFwd is not None:
                if useSlowMaps:
                    self.S1Pm[i] = self.angularSteps[i] / thtFwd[i]
                else:
                    self.F1Pm[i] = self.angularSteps[i] / thtFwd[i]
                head = calTable.find("Joint1_fwd_stepsizes").text.split(",")[:2]
                body = list(map(f2s, np.rad2deg(thtFwd[i])))
                calTable.find("Joint1_fwd_stepsizes").text = ','.join(head + body) + ','
            if thtRev is not None:
                if useSlowMaps:
                    self.S1Nm[i] = self.angularSteps[i] / thtRev[i]
                else:
                    self.F1Nm[i] = self.angularSteps[i] / thtRev[i]
                head = calTable.find("Joint1_rev_stepsizes").text.split(",")[:2]
                body = list(map(f2s, np.rad2deg(thtRev[i])))
                calTable.find("Joint1_rev_stepsizes").text = ','.join(head + body) + ','
            if phiFwd is not None:
                if useSlowMaps:
                    self.S2Pm[i] = self.angularSteps[i] / phiFwd[i]
                else:
                    self.F2Pm[i] = self.angularSteps[i] / phiFwd[i]
                head = calTable.find("Joint2_fwd_stepsizes").text.split(",")[:2]
                body = list(map(f2s, np.rad2deg(phiFwd[i])))
                calTable.find("Joint2_fwd_stepsizes").text = ','.join(head + body) + ','
            if phiRev is not None:
                if useSlowMaps:
                    self.S2Nm[i] = self.angularSteps[i] / phiRev[i]
                else:
                    self.F2Nm[i] = self.angularSteps[i] / phiRev[i]
                head = calTable.find("Joint2_rev_stepsizes").text.split(",")[:2]
                body = list(map(f2s, np.rad2deg(phiRev[i])))
                calTable.find("Joint2_rev_stepsizes").text = ','.join(head + body) + ','

    def XXupdateMotorFrequency(self, theta=None, phi=None):
        """Update cobra motor frequency

        Parameters
        ----------
        theta: object
            A numpy array with the theta motor frequency.
        phi: object
            A numpy array with the phi motor frequency.

        """

        if theta is not None and len(theta) != self.nCobras:
            raise RuntimeError("number of theta motor frequency must match number of cobras")
        if phi is not None and len(phi) != self.nCobras:
            raise RuntimeError("number of phi motor frequency must match number of cobras")

        for i in range(self.nCobras):
            header = self.dataContainers[i].find("DATA_HEADER")
            if theta is not None:
                self.motorFreq1[i] = theta[i]
                header.find('Motor1_Run_Frequency').text = str(theta[i])
            if phi is not None:
                self.motorFreq2[i] = phi[i]
                header.find('Motor2_Run_Frequency').text = str(phi[i])

    def updateMotorFrequency(self, theta=None, phi=None, moduleId=None, cobraId=None):
        """Update cobra motor frequency

        Parameters
        ----------
        theta: object
            A numpy array with the theta motor frequency.
        phi: object
            A numpy array with the phi motor frequency.
        moduleId: int
            If set, the module for the cobras
        cobraId: int
            If set, the per-module id for the (single) cobra
            Note that moduleId must also be set.

        We want to allow updating individual cobra, board, or modules
        """

        # Normalize lengths
        if theta is None and phi is None:
            return

        if moduleId is not None:
            if cobraId is not None:
                idx = [self.findCobraByModuleAndPositioner(moduleId, cobraId)]
            else:
                idx = self.findCobrasForModule(moduleId)
        else:
            if cobraId is not None:
                raise ValueError("if cobraId is specified, moduleId must also be.")
            idx = range(self.nCobras)

        # Allow passing in values.
        if theta is None or np.isscalar(theta):
            theta = [theta]*len(idx)
        if phi is None or np.isscalar(phi):
            phi = [phi]*len(idx)

        if len(phi) != len(theta):
            raise ValueError(f"length of phi and theta arrays must match. Found {len(phi)} and {len(theta)}")

        if len(theta) != len(idx):
            raise RuntimeError(
                f"number of motor frequencies ({len(theta)}) must match number of cobras ({len(idx)})")

        for i_i, i in enumerate(idx):
            header = self.dataContainers[i].find("DATA_HEADER")
            if theta[i_i] is not None:
                self.motorFreq1[i] = theta[i_i]
                header.find('Motor1_Run_Frequency').text = str(theta[i_i])
            if phi is not None:
                self.motorFreq2[i] = phi[i_i]
                header.find('Motor2_Run_Frequency').text = str(phi[i_i])

    def updateGeometry(self, centers=None, thetaArms=None, phiArms=None):
        """Update cobra centres.

        Parameters
        ----------
        centers: object
            A complex numpy array with the cobras central positions.
        thetaArms: object
            A numpy array with the cobras theta arm lengths.
        phiArms: object
            A numpy array with the cobras phi arm lengths.

        """

        if centers is not None and len(centers) != self.nCobras:
            raise RuntimeError("number of cobra centers must match number of cobras")
        if thetaArms is not None and len(thetaArms) != self.nCobras:
            raise RuntimeError("number of theta arm lengths must match number of cobras")
        if phiArms is not None and len(phiArms) != self.nCobras:
            raise RuntimeError("number of phi arm lengths must match number of cobras")

        for i in range(self.nCobras):
            kinematics = self.dataContainers[i].find("KINEMATICS")
            if centers is not None:
                self.centers[i] = centers[i]
                kinematics.find("Global_base_pos_x").text = str(centers[i].real)
                kinematics.find("Global_base_pos_y").text = str(centers[i].imag)
            if thetaArms is not None:
                self.L1[i] = thetaArms[i]
                kinematics.find("Link1_Link_Length").text = str(thetaArms[i])
            if phiArms is not None:
                self.L2[i] = phiArms[i]
                kinematics.find("Link2_Link_Length").text = str(phiArms[i])

    def updateThetaHardStops(self, ccw=None, cw=None):
        """Update cobra theta hard stop angles

        Parameters
        ----------
        ccw: object
            A numpy array with the cobras theta CCW hard stop angles.
        cw: object
            A numpy array with the cobras theta CW hard stop angles.

        """

        if ccw is not None and len(ccw) != self.nCobras:
            raise RuntimeError("number of cobra CCW hard stops must match number of cobras")
        if cw is not None and len(cw) != self.nCobras:
            raise RuntimeError("number of cobra CW hard stops must match number of cobras")

        for i in range(self.nCobras):
            kinematics = self.dataContainers[i].find("KINEMATICS")
            if ccw is not None:
                if not np.isfinite(ccw[i]):
                    raise ValueError(f"nan/inf in CCW limit for cobra idx {i}")
                self.tht0[i] = ccw[i]
                kinematics.find("CCW_Global_base_ori_z").text = str(np.rad2deg(ccw[i]))
            if cw is not None:
                if not np.isfinite(cw[i]):
                    raise ValueError(f"nan/inf in CW limit for cobra idx {i}")
                self.tht1[i] = cw[i]
                kinematics.find("CW_Global_base_ori_z").text = str(np.rad2deg(cw[i]))

    def updatePhiHardStops(self, ccw=None, cw=None):
        """Update cobra phi hard stop angles

        Parameters
        ----------
        ccw: object
            A numpy array with the cobras phi CCW hard stop angles.
        cw: object
            A numpy array with the cobras phi CW hard stop angles.

        """

        if ccw is not None and len(ccw) != self.nCobras:
            raise RuntimeError("number of cobra CCW hard stops must match number of cobras")
        if cw is not None and len(cw) != self.nCobras:
            raise RuntimeError("number of cobra CW hard stops must match number of cobras")

        for i in range(self.nCobras):
            kinematics = self.dataContainers[i].find("KINEMATICS")
            if ccw is not None:
                self.phiIn[i] = ccw[i] - np.pi
                kinematics.find("Joint2_CCW_limit_angle").text = str(np.rad2deg(ccw[i]))
            if cw is not None:
                self.phiOut[i] = cw[i] - np.pi
                kinematics.find("Joint2_CW_limit_angle").text = str(np.rad2deg(cw[i]))

    def updateOntimes(self, thetaFwd=None, thetaRev=None, phiFwd=None, phiRev=None, fast=True):
        """Update cobra ontimes

        Parameters
        ----------
        thetaFwd: object
            A numpy array with the cobras theta forward ontimes.
        thetaRev: object
            A numpy array with the cobras theta reverse ontimes.
        phiFwd: object
            A numpy array with the cobras phi forward ontimes.
        phiRev: object
            A numpy array with the cobras phi reverse ontimes.
        fast: boolean
            Update fast or slow motor maps.

        """

        if thetaFwd is not None and len(thetaFwd) != self.nCobras:
            raise RuntimeError("number of cobra theta forward ontimes must match number of cobras")
        if thetaRev is not None and len(thetaRev) != self.nCobras:
            raise RuntimeError("number of cobra theta reverse ontimes must match number of cobras")
        if phiFwd is not None and len(phiFwd) != self.nCobras:
            raise RuntimeError("number of cobra phi forward ontimes must match number of cobras")
        if phiRev is not None and len(phiRev) != self.nCobras:
            raise RuntimeError("number of cobra phi reverse ontimes must match number of cobras")

        for i in range(self.nCobras):
            kinematics = self.dataContainers[i].find("KINEMATICS")
            if fast:
                if thetaFwd is not None:
                    self.motorOntimeFwd1[i] = thetaFwd[i]
                    kinematics.find("Link1_fwd_Duration").text = str(thetaFwd[i])
                if thetaRev is not None:
                    self.motorOntimeRev1[i] = thetaRev[i]
                    kinematics.find("Link1_rev_Duration").text = str(thetaRev[i])
                if phiFwd is not None:
                    self.motorOntimeFwd2[i] = phiFwd[i]
                    kinematics.find("Link2_fwd_Duration").text = str(phiFwd[i])
                if phiRev is not None:
                    self.motorOntimeRev2[i] = phiRev[i]
                    kinematics.find("Link2_rev_Duration").text = str(phiRev[i])
            else:
                if thetaFwd is not None:
                    self.motorOntimeSlowFwd1[i] = thetaFwd[i]
                    kinematics.find("Link1_fwd_Duration_Slow").text = str(thetaFwd[i])
                if thetaRev is not None:
                    self.motorOntimeSlowRev1[i] = thetaRev[i]
                    kinematics.find("Link1_rev_Duration_Slow").text = str(thetaRev[i])
                if phiFwd is not None:
                    self.motorOntimeSlowFwd2[i] = phiFwd[i]
                    kinematics.find("Link2_fwd_Duration_Slow").text = str(phiFwd[i])
                if phiRev is not None:
                    self.motorOntimeSlowRev2[i] = phiRev[i]
                    kinematics.find("Link2_rev_Duration_Slow").text = str(phiRev[i])

    def createCalibrationFile(self, outputFileName, name=None, site=None):
        """Creates a new XML calibration file based on current configuration

        Parameters
        ----------
        outputFileName: object
            The path where the output XML calibration file should be saved.
        name : str
            A string to put into a top_level ARM_DATA_NAME element
        site : str
            A string to put into a top_level ARM_DATA_SITE element
        """

        # Create the output XML tree
        newXmlTree = ElementTree.ElementTree(ElementTree.Element("ARM_DATA"))
        newRootElement = newXmlTree.getroot()
        if name is not None:
            node = ElementTree.Element("ARM_DATA_NAME")
            node.text = name
            newRootElement.append(node)

        if site is not None:
            node = ElementTree.Element("ARM_DATA_SITE")
            node.text = site
            newRootElement.append(node)

        # Fill the calibration file
        for i in range(self.nCobras):
            # Append the arm data container to the root element
            newRootElement.append(self.dataContainers[i])

        # Save the new XML calibration file
        newXmlTree.write(outputFileName, encoding="UTF-8", xml_declaration=True)
        self.logger.info(
            f'wrote pfiDesign file for {self.nCobras} cobras and name={name} to {str(outputFileName)}')

    def validatePhiLimits(self, rangeOnly=True):
        """ Confirm that the phi limits are sane. """

        phiRange = self.phiOut - self.phiIn
        phiRange[phiRange < 0] += 2*np.pi
        phiRange[phiRange >= 2*np.pi] -= 2*np.pi

        if not rangeOnly:
            raise NotImplementedError('not checking phi limit _positions_ yet.')

        duds = phiRange < np.pi

        for cidx in np.where(duds)[0]:
            with np.printoptions(precision=2, suppress=True):
                self.logger.warn(f'phi limits bad: mod={self.moduleIds[cidx]} pos={self.positionerIds[cidx]} '
                                 f'CCW={np.rad2deg(self.phiIn[cidx])} CW={np.rad2deg(self.phiOut[cidx])}')
        return ~duds

    def validateThetaLimits(self):
        """ Confirm that the theta limits are sane. """

        thetaRange = self.tht1 - self.tht0
        thetaRange[thetaRange < 0] += 2*np.pi
        thetaRange[thetaRange < np.pi/4] += 2*np.pi
        duds = thetaRange < np.deg2rad(370)

        for cidx in np.where(duds)[0]:
            with np.printoptions(precision=2, suppress=True):
                self.logger.warn(f'theta limits bad: mod={self.moduleIds[cidx]} pos={self.positionerIds[cidx]} '
                                 f'range={np.rad2deg(thetaRange[cidx])}')
        return ~duds
