#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

HOME = [
    0.8633871078491211,
    0.42438486218452454,
    0.18764179944992065,
    -1.3867534399032593,
    -0.05604249984025955,
    1.7663707733154297,
    1.8812178373336792,
]

KIT = [
    -0.22613531351089478,
    -0.052150316536426544,
    0.3750353157520294,
    -1.90276300907135,
    0.03822034224867821,
    1.83684504032135,
    2.5277857780456543,
]

class GelloPublisher(Node):
    def __init__(self):
        super().__init__('gello_joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/gello/joint_states', 10)
        self.timer = self.create_timer(0.01, self.publish)  # 100Hz
        self.get_logger().info('Publishing init pose at 100Hz...')
        
        # self.publish(KIT)
        # self.get_logger().info('Publishing pose ONCE')
        
        

    def publish(self, pose=HOME):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3',
            'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        msg.position = HOME
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = GelloPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()