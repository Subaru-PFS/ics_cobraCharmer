"""

Modified from ics.cobraOps.CobrasCalibrationProduct
The values here are not modified from the XML file, so there
is no pixel scaling or phi home hacking. The coordinate
system here should be in F3C.

"""

import numpy as np
import xml.etree.ElementTree as ElementTree
from copy import deepcopy

class PFIDesign():
    """

    Class describing a cobras calibration product.

    """

    def __init__(self, fileName):
        """Constructs a new cobras calibration product using the information
        contained in an XML calibration file.

        Parameters
        ----------
        fileName: object
            The path to the XML calibration file.

        Returns
        -------
        object
            The cobras calibration product.

        """
        # Load the XML calibration file
        calibrationFileRootElement = ElementTree.parse(fileName).getroot()

        # Get all the data container elements
        dataContainers = calibrationFileRootElement.findall("ARM_DATA_CONTAINER")
        self.origin_dataContainers = dataContainers
        self.dataContainers = deepcopy(dataContainers)

        # The number of cobras is equal to the number of data containers
        self.nCobras = len(dataContainers)

        # Create some of the calibration data arrays
        self.moduleIds = np.empty(self.nCobras, dtype="int")
        self.positionerIds = np.empty(self.nCobras, dtype="int")
        self.serialIds = np.empty(self.nCobras, dtype="int")
        self.centers = np.empty(self.nCobras, dtype="complex")
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

        self.motorOfftimeFwd1 = np.empty(self.nCobras)
        self.motorOfftimeFwd2 = np.empty(self.nCobras)
        self.motorOfftimeRev1 = np.empty(self.nCobras)
        self.motorOfftimeRev2 = np.empty(self.nCobras)

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

            # Save some of the kinematics information
            kinematics = dataContainers[i].find("KINEMATICS")
            self.centers[i] = float(kinematics.find("Global_base_pos_x").text) + float(kinematics.find("Global_base_pos_y").text) * 1j
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

                self.motorOfftimeFwd1[i] = float(kinematics.find('Link1_fwd_Intervals').text)
                self.motorOfftimeRev1[i] = float(kinematics.find('Link1_rev_Intervals').text)
                self.motorOfftimeFwd2[i] = float(kinematics.find('Link2_fwd_Intervals').text)
                self.motorOfftimeRev2[i] = float(kinematics.find('Link2_rev_Intervals').text)

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
        return np.where(self.moduleIds == moduleId)[0]

    def findAllCobras(self):
        return range(self.nCobras)

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

    def setModuleId(self, moduleId, forModule=None):
        """Update moduleIds

        Args
        ----
        moduleId: int
            The moduleId to set ourselves to.
        forModule: int
            If set, the module within ourselves to set.
        """

        if forModule is not None:
            idx = self.findCobrasForModule(forModule)
        else:
            idx = range(self.nCobras)

        # A length test is probably sufficient
        if len(idx) != 57:
            raise RuntimeError("Will not set moduleId to anything other that all cobras in a module")
        for i in idx:
            self.moduleIds[i] = moduleId

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
        if theta is None:
            if phi is None:
                return
            theta = [None]*len(phi)
        elif phi is None:
            phi = [None]*len(theta)

        if len(phi) != len(theta):
            raise ValueError(f"length of phi and theta arrays must match. Found {len(phi)} and {len(theta)}")

        if moduleId is not None:
            if cobraId is not None:
                idx = [self.findCobraByModuleAndPositioner(moduleId, cobraId)]
            else:
                idx = self.findCobrasForModule(moduleId)
        else:
            if cobraId is not None:
                raise ValueError("if cobraId is specified, moduleId must also be.")
            idx = range(self.nCobras)

        if len(theta) != len(idx):
            raise RuntimeError(f"number of motor frequencies ({len(theta)}) must match number of cobras ({len(idx)})")

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
                self.tht0[i] = ccw[i]
                kinematics.find("CCW_Global_base_ori_z").text = str(np.rad2deg(ccw[i]))
            if cw is not None:
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

    def updateOntimes(self, thtFwd=None, thtRev=None, phiFwd=None, phiRev=None):
        """Update cobra ontimes

        Parameters
        ----------
        thtFwd: object
            A numpy array with the cobras theta forward ontimes.
        thtRev: object
            A numpy array with the cobras theta reverse ontimes.
        phiFwd: object
            A numpy array with the cobras phi forward ontimes.
        phiRev: object
            A numpy array with the cobras phi reverse ontimes.

        """

        if thtFwd is not None and len(thtFwd) != self.nCobras:
            raise RuntimeError("number of cobra theta forward ontimes must match number of cobras")
        if thtRev is not None and len(thtRev) != self.nCobras:
            raise RuntimeError("number of cobra theta reverse ontimes must match number of cobras")
        if phiFwd is not None and len(phiFwd) != self.nCobras:
            raise RuntimeError("number of cobra phi forward ontimes must match number of cobras")
        if phiRev is not None and len(phiRev) != self.nCobras:
            raise RuntimeError("number of cobra phi reverse ontimes must match number of cobras")

        for i in range(self.nCobras):
            kinematics = self.dataContainers[i].find("KINEMATICS")
            if thtFwd is not None:
                self.motorOntimeFwd1[i] = thtFwd[i]
                kinematics.find("Link1_fwd_Duration").text = str(thtFwd[i])
            if thtRev is not None:
                self.motorOntimeRev1[i] = thtRev[i]
                kinematics.find("Link1_rev_Duration").text = str(thtRev[i])
            if phiFwd is not None:
                self.motorOntimeFwd2[i] = phiFwd[i]
                kinematics.find("Link2_fwd_Duration").text = str(phiFwd[i])
            if phiRev is not None:
                self.motorOntimeRev2[i] = phiRev[i]
                kinematics.find("Link2_rev_Duration").text = str(phiRev[i])

    def createCalibrationFile(self, outputFileName):
        """Creates a new XML calibration file based on current configuration

        Parameters
        ----------
        outputFileName: object
            The path where the output XML calibration file should be saved.

        """

        # Create the output XML tree
        newXmlTree = ElementTree.ElementTree(ElementTree.Element("ARM_DATA"))
        newRootElement = newXmlTree.getroot()

        # Fill the calibration file
        for i in range(self.nCobras):
            # Append the arm data container to the root element
            newRootElement.append(self.dataContainers[i])

        # Save the new XML calibration file
        newXmlTree.write(outputFileName, encoding="UTF-8", xml_declaration=True)
