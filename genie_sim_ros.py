from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import (
    CompressedImage,
    JointState,
)
from collections import deque
import threading

import numpy as np

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class SimROSNode(Node):
    def __init__(self, node_name="univla_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        # publish
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/sim/target_joint_state",
            QOS_PROFILE_LATEST,
        )

        # subscribe
        self.sub_img_head = self.create_subscription(
            CompressedImage,
            "/sim/head_img",
            self.callback_rgb_image_head,
            1,
        )

        self.sub_img_left_wrist = self.create_subscription(
            CompressedImage,
            "/sim/left_wrist_img",
            self.callback_rgb_image_left_wrist,
            1,
        )

        self.sub_img_right_wrist = self.create_subscription(
            CompressedImage,
            "/sim/right_wrist_img",
            self.callback_rgb_image_right_wrist,
            1,
        )

        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            1,
        )

        # msg
        self.lock_img_head = threading.Lock()
        self.lock_img_left_wrist = threading.Lock()
        self.lock_img_right_wrist = threading.Lock()

        self.message_buffer = deque(maxlen=30)
        self.lock_joint_state = threading.Lock()
        self.obs_joint_state = JointState()
        self.cur_joint_state = JointState()

        # loop
        self.loop_rate = self.create_rate(30.0)

        self.img_head = None
        self.img_left_wrist = None
        self.img_right_wrist = None

    def callback_rgb_image_head(self, msg):
        # print(msg.header)
        with self.lock_img_head:
            self.img_head = msg

    def callback_rgb_image_left_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_left_wrist:
            self.img_left_wrist = msg

    def callback_rgb_image_right_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_right_wrist:
            self.img_right_wrist = msg

    def get_img_head(self):
        with self.lock_img_head:
            return self.img_head

    def get_img_left_wrist(self):
        with self.lock_img_left_wrist:
            return self.img_left_wrist

    def get_img_right_wrist(self):
        with self.lock_img_right_wrist:
            return self.img_right_wrist

    def publish_joint_command(self, action):

        cmd_msg = JointState()
        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        ]
        cmd_msg.position = [0.0] * len(cmd_msg.name)
        cmd_msg.position[0] = action[0]
        cmd_msg.position[1] = action[1]
        cmd_msg.position[2] = action[2]
        cmd_msg.position[3] = action[3]
        cmd_msg.position[4] = action[4]
        cmd_msg.position[5] = action[5]
        cmd_msg.position[6] = action[6]
        cmd_msg.position[7] = np.clip((1 - action[7]), 0, 1)
        cmd_msg.position[8] = action[8]
        cmd_msg.position[9] = action[9]
        cmd_msg.position[10] = action[10]
        cmd_msg.position[11] = action[11]
        cmd_msg.position[12] = action[12]
        cmd_msg.position[13] = action[13]
        cmd_msg.position[14] = action[14]
        cmd_msg.position[15] = np.clip((1 - action[15]), 0, 1)

        self.pub_joint_command.publish(cmd_msg)

    def callback_joint_state(self, msg):
        # print(msg.header)
        self.cur_joint_state = msg

        joint_name_state_dict = {}
        for idx, name in enumerate(msg.name):
            joint_name_state_dict[name] = msg.position[idx]

        msg_remap = JointState()
        msg_remap.header = msg.header
        msg_remap.name = []
        msg_remap.velocity = []
        msg_remap.effort = []
        msg_remap.position.append(joint_name_state_dict["idx21_arm_l_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx22_arm_l_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx23_arm_l_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx24_arm_l_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx25_arm_l_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx26_arm_l_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx27_arm_l_joint7"])
        left_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx41_gripper_l_outer_joint1"]))))
        msg_remap.position.append(left_gripper_pos)

        msg_remap.position.append(joint_name_state_dict["idx61_arm_r_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx62_arm_r_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx63_arm_r_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx64_arm_r_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx65_arm_r_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx66_arm_r_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx67_arm_r_joint7"])
        right_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx81_gripper_r_outer_joint1"]))))
        msg_remap.position.append(right_gripper_pos)

        with self.lock_joint_state:
            self.obs_joint_state = msg_remap

    def get_joint_state(self):
        with self.lock_joint_state:
            return self.obs_joint_state

