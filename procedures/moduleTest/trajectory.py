import numpy as np


class Trajectories():
    """Class that stores cobras trajectories.
    A trajectory is a list of cobra movements, so it could describe the complete
    trajectory until target convergence.
    """

    def __init__(self, nCobras, timeStep=10):
        """Constructs a new Trajectories instance.
        Parameters
        ----------
        nCobras: int
            The total number of cobras.
        timeStep: integer, optional
            The trajectories time step resolution in steps. Default is 10
            steps.
        Returns
        -------
        object
            The Trajectories instance.
        """
        # Store the input parameters
        self.nCobras = nCobras
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
            The trajectories time step resolution in step units.
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
        # Create the arrays for the new movement information
        movementSteps = thetaAngles.shape[1]
        newThetaAngles = np.empty((self.nCobras, movementSteps))
        newPhiAngles = np.empty((self.nCobras, movementSteps))

        # update positions for moving cobras
        newThetaAngles[cIds] = thetaAngles
        newPhiAngles[cIds] = phiAngles

        # copy last positions for non moving cobras
        if len(self.movements) > 0:
            notMoving = np.full(self.nCobras, True)
            notMoving[cIds] = False
            newThetaAngles[notMoving] = (self.movements[-1][0][notMoving, -1])[:, np.newaxis]
            newPhiAngles[notMoving] = (self.movements[-1][1][notMoving, -1])[:, np.newaxis]

        # Add the movement information to the list
        self.movements.append([newThetaAngles, newPhiAngles])

        # Increase the time step counter
        self.steps += movementSteps

        # Set to None the fiber and elbow position arrays, since they are now
        # outdated
        self._fiberPositions = None
        self._elbowPositions = None

    def calculateFiberPositions(self, cobraCoach):
        """Calculates the fiber positions along the cobras trajectories.
        Parameters
        ----------
        cobrasCroach: object
            A cobra coach instance.
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
        self._fiberPositions = np.empty((self.nCobras, self.steps), dtype=np.complex)

        # Calculate the fiber positions for each cobra movement
        lastStep = 0

        for movement in self.movements:
            thetaSteps = movement[0]
            phiSteps = movement[1]
            movementSteps = thetaSteps.shape[1]
            self._fiberPositions[:, lastStep:lastStep + movementSteps] = cobraCoach.pfi.anglesToPositions(
                cobraCoach.allCobras, thetaSteps, phiSteps)
            lastStep += movementSteps

        return self._fiberPositions

    def calculateElbowPositions(self, cobraCoach):
        """Calculates the elbow positions along the cobras trajectories.
        Parameters
        ----------
        cobrasCroach: object
            A cobra coach instance.
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
        self._elbowPositions = np.empty((self.nCobras, self.steps), dtype=np.complex)

        # Calculate the elbow positions for each cobra movement
        lastStep = 0

        for movement in self.movements:
            thetaSteps = movement[0]
            movementSteps = thetaSteps.shape[1]
            self._elbowPositions[:, lastStep:lastStep + movementSteps] = cobraCoach.pfi.anglesToElbowPositions(
                cobraCoach.allCobras, thetaSteps)
            lastStep += movementSteps

        return self._elbowPositions
