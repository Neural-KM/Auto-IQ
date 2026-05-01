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
        ],
    },
)
