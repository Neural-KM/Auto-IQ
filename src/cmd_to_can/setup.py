import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cmd_to_can'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skully',
    maintainer_email='boliyadeep2003@gmail.com',
    description='Bridge VehicleControl ROS messages to the Teensy 4.1 over CAN.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cmd_to_can_node = cmd_to_can.cmd_to_can_node:main',
            'vehicle_control_tui = cmd_to_can.vehicle_control_tui_node:main',
            # Standalone (non-ROS) pygame UI for live PID-gain tuning over
            # CAN IDs 0x105 (set) and 0x106 (feedback). Runs fine via
            # `ros2 run cmd_to_can pid_tuner_gui ...` or directly with
            # `python3 .../pid_tuner_gui.py`.
            'pid_tuner_gui = cmd_to_can.pid_tuner_gui:main',
        ],
    },
)
