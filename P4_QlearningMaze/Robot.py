import random


class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

        for x in range(self.maze.maze_data.shape[0]):
            for y in range(self.maze.maze_data.shape[1]):
                self.Qtable[(x, y)] = {}
                for action in self.valid_actions:
                    self.Qtable[(x, y)][action] = 0

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = self.epsilon0
        else:
            # TODO 2. Update parameters when learning
            self.t += 1
            self.epsilon = self.epsilon ** self.t

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.

        if state not in self.Qtable:
            self.Qtable[state] = {}
            for current_action in self.valid_actions:
                self.Qtable[state][current_action] = self.maze.move_robot(
                    current_action)

        return self.Qtable

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon

            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                self.action = random.choice(self.valid_actions)
                return self.action
            else:
                # TODO 7. Return action with highest q value
                self.action = sorted(self.Qtable[self.state],
                                     key=lambda x: self.Qtable[self.state][x],
                                     reverse=True)[0]
                return self.action
        elif self.testing:
            # TODO 7. choose action with highest q value
            self.action = sorted(self.Qtable[self.state],
                                 key=lambda x: self.Qtable[self.state][x],
                                 reverse=True)[0]
            return self.action
        else:
            # TODO 6. Return random choose aciton
            self.action = random.choice(self.valid_actions)
            return self.action

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # TODO 8. When learning, update the q table according
            # to the given rules
            future_rewards = []

            for next_action in self.valid_actions:
                future_rewards.append(self.Qtable[next_state][next_action])
            update_value = r + self.gamma * \
                max(future_rewards)

            self.Qtable[self.state][action] = (1 - self.alpha) * \
                self.Qtable[self.state][action] + self.alpha * update_value


        return self.Qtable

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state()  # Get the current state
        # For the state, create q table line
        self.create_Qtable_line(self.state)

        action = self.choose_action()  # choose action for this state
        reward = self.maze.move_robot(action)  # move robot for given action

        next_state = self.sense_state()  # get next state
        # create q table line for next state
        self.create_Qtable_line(next_state)

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state)  # update q table
            self.update_parameter()  # update parameters

        return action, reward
