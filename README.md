# q_learning_project
Implementation Plan
Names: Seha Choi and Shengjie Lin

# Q-learning algorithm
# Executing the Q-learning algorithm
We will just use the function we learned in class. For rewards we will put a positive reward for correct block placement at the AR code and a negative reward for the incorrect block placement. While the robot is in transition there will be 0 reward. We will test this by running a couple iterations and checking if they are the correct values.
# Determining when the Q-matrix has converged
We will say the Q-matrix is converged if the values for each action are within 0.02 difference (the value is subject to change after testing). We will test this via printing out the Q-Matrix values and see if they are consistent.
# Once the Q-matrix has converged, how to determine which actions the robot should take to maximize expected reward
We will take a greedy approach and pick the highest reward action. After the matrix has converged the actions should be pretty clear. I.e) Moving the block to the corresponding AR Code. We will verify that this is identical to taking a greedy approach.

Robot perception
Determining the identities and locations of the three colored objects
Determining the identities and locations of the three AR tags
Robot manipulation & movement
Picking up and putting down the colored objects with the OpenMANIPULATOR arm
Navigating to the appropriate locations to pick up and put down the colored objects
A brief timeline sketching out when you would like to have accomplished each of the components listed above.
