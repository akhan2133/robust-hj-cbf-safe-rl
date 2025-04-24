import rospy
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import TransformStamped
import tf
from time import sleep
import numpy as np
from threading import Lock
from redexp.brts.turtlebot_brt import (
    grid,
    turtlebot_2_no_model_mismatch,
    turtlebot_2_model_mismatch,
)

from redexp.config.turtlebot import (
    TASC_7001_X_BOUNDARY_LOWER,
    TASC_7001_X_BOUNDARY_UPPER,
    TASC_7001_Y_BOUNDARY_LOWER,
    TASC_7001_Y_BOUNDARY_UPPER,
    RADIUS,
    OBSTACLE_RADIUS,
)

VICON_TOPIC = "/vicon/ml_turtlebot_2/turtlebot_2"
VALUE_TOPIC = "/turtlebot/value"

ROTATION_OFFSET = -np.pi / 32
X_OFFSET = +0.0
Y_OFFSET = +0.0

DEBUG = False


def calculate_heading(pose):
    x = pose.rotation.x
    y = pose.rotation.y
    z = pose.rotation.z
    w = pose.rotation.w
    quaternion = (x, y, z, w)

    # add offset to make yaw=0 face the computers
    rotation_quaternion = tf.transformations.quaternion_from_euler(
        0, 0, ROTATION_OFFSET
    )

    quaternion = tf.transformations.quaternion_multiply(rotation_quaternion, quaternion)

    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    return yaw


def update_state(ts_msg):
    pose = ts_msg.transform
    x = pose.translation.x
    y = pose.translation.y

    x += X_OFFSET
    y += Y_OFFSET
    # theta off-set done in heading calclation
    theta = calculate_heading(pose)

    return np.array([x, y, theta])


class Turtlebot:
    def __init__(self, goal_location, goal_r, model_mismatch) -> None:
        self.state = np.array([0.0, 0.0, 0.0])
        self.mutex = Lock()

        if model_mismatch:
            self.brt = np.load(
                "./redexp/brts/turtlebot_2_brt_speed_06_wMax_06_dstb.npy"
            )
            self.dyn = turtlebot_2_model_mismatch
        else:
            self.brt = np.load(
                "./redexp/brts/turtlebot_2_brt_speed_06_wMax_11_dstb.npy"
            )
            self.dyn = turtlebot_2_no_model_mismatch

        self.true_brt = np.load(
            "./redexp/brts/turtlebot_2_brt_speed_06_wMax_11_dstb.npy"
        )
        self.grid = grid

        self.goal_location = goal_location
        self.goal_r = goal_r

        rospy.init_node("turtlebot_controller_node")
        self.pub = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        rospy.Subscriber(
            VICON_TOPIC,
            TransformStamped,
            self.update_state,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("initialized turtlebot_controller_node")

    def update_state(self, ts_msg):
        with self.mutex:
            self.state = update_state(ts_msg)

    def get_state(self):
        with self.mutex:
            return self.state

    def set_action(self, action):
        if not self.in_bounds():
            print("TURTLEBOT2 OUT OF BOUNDS")
            vel_cmd = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0))
        elif self.reach_goal():
            print("TURTLEBOT2 REACHED GOAL")
            vel_cmd = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0))
        elif self.near_obs():
            print("TURTLEBOT2 TOO CLOSE TO OBSTACLE")
            vel_cmd = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0))
        else:
            vel_cmd = Twist(linear=Vector3(0.6, 0, 0), angular=Vector3(0, 0, action[0]))

        if DEBUG:
            value = grid.get_value(self.brt, self.get_state())
            print(f"DEBUG: {action=} {value=}")
        else:
            # add delay
            sleep(np.random.rand() / 4)
            self.pub.publish(vel_cmd)

    def in_bounds(self):
        (
            x,
            y,
            _,
        ) = self.get_state()
        return (
            TASC_7001_X_BOUNDARY_LOWER <= x <= TASC_7001_X_BOUNDARY_UPPER
            and TASC_7001_Y_BOUNDARY_LOWER <= y <= TASC_7001_Y_BOUNDARY_UPPER
        )

    def near_obs(self):
        state = self.get_state()
        reached_goal = np.linalg.norm(state[:2]) < (RADIUS + OBSTACLE_RADIUS)
        return reached_goal

    def reach_goal(self):
        state = self.get_state()
        reached_goal = (
            np.linalg.norm(state[:2] - self.goal_location) < self.dyn.r + self.goal_r
        )
        return reached_goal

    def get_brt_value(self):
        state = self.get_state()
        value = grid.get_value(self.true_brt, state)
        return value


if __name__ == "__main__":
    try:
        rospy.init_node("turtlebot_monitor_node", log_level=rospy.DEBUG)
        rospy.loginfo("Initialized turtlebot_monitor_node")

        def monitor_update_state(ts_msg):
            state = update_state(ts_msg)
            value = grid.get_value(
                np.load("./redexp/brts/turtlebot_2_brt_speed_06_wMax_11_dstb.npy"),
                state,
            )
            rospy.logdebug(f"turtlebot2 state: {state}")
            rospy.logdebug(f"value = {value}")

        rospy.Subscriber(
            VICON_TOPIC,
            TransformStamped,
            monitor_update_state,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("shutdown")
