import numpy as np


class Trajectories():
    """Class that stores cobras trajectories.
    A trajectory is a list of cobra movements, so it could describe the complete
    trajectory until target convergence.
    """

    def __init__(self, calibrationProduct, timeStep=10):
        """Constructs a new Trajectories instance.
        Parameters
        ----------
        calibrationProduct: object
            The cobras calibration product containing the cobra properties.
        timeStep: integer, optional
            The trajectories time step resolution in steps. Default is 10
            steps.
        Returns
        -------
        object
            The Trajectories instance.
        """
        # Store the input parameters
        self.calibrationProduct = calibrationProduct
        self.nCobras = calibrationProduct.nCobras
        self.timeStep = timeStep

        # Initialize the movements list
        self.movements = []

        # Initialize the time step counter
        self.steps = 0

        # Set to None the fiber and elbow positions arrays
        self._fiberPositions = None
        self._elbowPositions = None

    def getTimeStep(self):
        """Returns the trajectories time step resolution.
        Returns
        -------
        float
            The trajectories time step resolution in seconds.
        """
        return self.timeStep

    def addMovement(self, cIds, thetaAngles, phiAngles):
        """Adds a new cobra movement to the trajectories.
        Note that the theta and phi steps should have the same time step
        resolution as the trajectories.
        Parameters
        ----------
        cIds: integer array
            The index for the moving cobras
        thetaAngles: object
            A 2D numpy array with the movement theta angles to add to the
            trajectories.
        phiAngles: object
            A 2D numpy array with the movement phi angles to add to the
            trajectories.
        """

        # update positions for moving cobras
        size = thetaAngles.shape[1]
        newThetaAngles = np.zeros((self.nCobras, size))
        newPhiAngles = np.zeros((self.nCobras, size))

        for c in range(len(cIds)):
            newThetaAngles[cIds[c],:] = thetaAngles[c]
            newPhiAngles[cIds[c],:] = phiAngles[c]

        # copy last positions for non moving cobras
        if self.steps > 0:
            for c in range(self.nCobras):
                if c not in cIds:
                    newThetaAngles[c,:] = self.movements[-1][0][c,-1]
                    newPhiAngles[c,:] = self.movements[-1][1][c,-1]

        # Add the movement information to the list
        self.movements.append([newThetaAngles, newPhiAngles])

        # Increase the time step counter
        self.steps += size

        # Set to None the fiber and elbow position arrays, since they are now
        # outdated
        self._fiberPositions = None
        self._elbowPositions = None

    def calculateFiberPositions(self):
        """Calculates the fiber positions along the cobras trajectories.
        Returns
        -------
        object
            A complex 2D numpy array with the fiber positions along the cobras
            trajectories. None if the trajectories have no cobra movements.
        """
        # Return None if the trajectories have no cobra movements
        if len(self.movements) == 0:
            return None

        # Return the fiber positions if they have already been calculated
        if self._fiberPositions is not None:
            return self._fiberPositions

        # Initialize the fiber positions array
        cobras = self.movements[0][0].shape[0]
        self._fiberPositions = np.empty((cobras, self.steps))

        # Calculate the fiber positions for each cobra movement
        centers = self.calibrationProduct.centers
        L1 = self.calibrationProduct.L1
        L2 = self.calibrationProduct.L2
        lastStep = 0

        for movement in self.movements:
            thetaSteps = movement[0]
            phiSteps = movement[1]
            movementSteps = len(thetaSteps.shape[1])
            self._fiberPositions[:, lastStep:movementSteps] = centers + L1 * np.exp(
                1j * thetaSteps) + L2 * np.exp(1j * (thetaSteps + phiSteps))
            lastStep += movementSteps

        return self._fiberPositions

    def calculateElbowPositions(self):
        """Calculates the elbow positions along the cobras trajectories.
        Returns
        -------
        object
            A complex 2D numpy array with the elbow positions along the cobras
            trajectories. None if the trajectories have no cobra movements.
        """
        # Return None if the trajectories have no cobra movements
        if len(self.movements) == 0:
            return None

        # Return the elbow positions if they have already been calculated
        if self._elbowPositions is not None:
            return self._elbowPositions

        # Initialize the elbow positions array
        cobras = self.movements[0][0].shape[0]
        self._elbowPositions = np.empty((cobras, self.steps))

        # Calculate the elbow positions for each cobra movement
        centers = self.calibrationProduct.centers
        L1 = self.calibrationProduct.L1
        lastStep = 0

        for movement in self.movements:
            thetaSteps = movement[0]
            movementSteps = len(thetaSteps.shape[1])
            self._elbowPositions[:, lastStep:movementSteps] = centers + L1 * np.exp(
                1j * thetaSteps)
            lastStep += movementSteps

        return self._elbowPositions

    def simulateMCSimage(self, step):
        """Simulates an MCS image at the given trajectories time step position.
        Parameters
        ----------
        step: int
            The trajectories time step where the MCS image should be simulated.
        Returns
        -------
        object
            The simulated MCSImage.
        """
        # Calculate the fiber positions at the given time step
        fiberPositions = self.calculateFiberPositions()[:, step]

        # This needs to be implemented using some code from Jennifer
        return simulateImage(fiberPositions)

    def detectCobraCollisions(self):
        """Detects cobra collision using the trajectories information.
        """
        # Calculate the fiber and elbow positions
        fiberPositions = self.calculateFiberPositions()
        elbowPositions = self.calculateElbowPositions()

        # This needs to be implemented using some code from ics_cobraOps
        return detectCollisions(fiberPositions, elbowPositions)

