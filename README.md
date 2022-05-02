# Q-Learning Project
Team members: Seha Choi and Shengjie Lin
## Implementation Plan
### Q-learning algorithm
* Executing the Q-learning algorithm

    We will just use the function we learned in class. For rewards we will put a positive reward for correct block placement at the AR code and a negative reward for the incorrect block placement. While the robot is in transition there will be 0 reward. We will test this by running a couple iterations and checking if they are the correct values.
* Determining when the Q-matrix has converged

     We will say the Q-matrix is converged if the values for each action are within 0.02 difference (the value is subject to change after testing). We will test this via printing out the Q-Matrix values and see if they are consistent.
* Once the Q-matrix has converged, how to determine which actions the robot should take to maximize expected reward

    We will take a greedy approach and pick the highest reward action. After the matrix has converged the actions should be pretty clear. I.e) Moving the block to the corresponding AR Code. We will verify that this is identical to taking a greedy approach.
### Robot perception
* Determining the identities and locations of the three colored objects

    We will process the image frames by computing a mask of the pixels corresponding to the specified color. Then we will find its center by computing the image moment. That will be the location of the colored object. To veriy this, we will visualize the center of the color patch.
* Determining the identities and locations of the three AR tags

    To detect AR tags, we will process the image frames uisng the `cv2.aruco` module. It will give us the ids and corner locations of the AR tags. To verify this, we will visualize the borders of the AR tags.
### Robot manipulation & movement
* Picking up and putting down the colored objects with the OpenMANIPULATOR arm

    Once the robot is ready at the appropriate location, we will use `MoveIt` package to manipulate the robot arm to pick up the object. To verify this, we will check whether the manipulation is successful in the simulator.
* Navigating to the appropriate locations to pick up and put down the colored objects

     We will let the turtlebot self-rotate until it sees pixels of the specified color. Then we will let it move forward to a certain distance from the object while steering itself by trying to center the colored patch, both of which are achieved by using a p-controller. To verify this, we will check whether the robot exhibits the desired behavior in simulation.
### TimeLine
Q-Learning Algorithm - By May 1st. We will finish the Q_Learning algorithm part by May 1st and see if the matrix converges to the expected values.

Robot Perception - By May 4th. Since the Q-Learning Algorithm is separate from the Robot movements we should be able to test independently that the robot is able to execute actions via perception.

Robot Manipulation & Movement - By May 4th. Similar to above since this is separate from the Q-Learning algorithm we could work in parallel and divide the code up.

Final wrap up - By May 8th. We will see if the matrix converges correctly and if the robot is able to execute the commands generated by the matrix correctly.
## Write up
### High-level description
We first use the Q-Learning Algorithm to determine which object belongs to which tag. Then with that knowledge we train the robot to actually execute the movement of dumbells to the corresponding AR tag.
### Q-Learning Algorithm
We first initialize an empty Q matrix and set all values to 0. We then populate invalid state action pairs to a value of -1. In addition, we create a transition matrix which tells us which state an action will lead to. i.e) transtiion matrix[state, action] = new state
### Update Q Matrix
We just follow the Q-learning algorithm and update the corresponding values using the reward that is received after performing a valid action. For selecting action we just uniformly select an action among the valid actions.
### Testing Convergenge
We test to see if the values in the Q-Matrix does not change after a certain amount of movements. A Q value is regarded as unchanged if the difference is less than a threshold of 0.5.
