from setuptools import setup, find_packages

package_name = 'neural_slam_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam',
    maintainer_email='sam@example.com',
    description='Neural SLAM ROS2 package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'neural_recon_node = neural_slam_ros.neural_recon_node:main',
        ],
    },
)