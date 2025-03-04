from setuptools import setup
import os
from glob import glob

package_name = 'neural_slam_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tom',
    maintainer_email='tom@example.com',
    description='Neural SLAM ROS2 integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'neural_recon_node.py = neural_slam_ros.neural_recon_node:main',
        ],
    },
)