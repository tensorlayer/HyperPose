from setuptools import setup,find_packages

description_file=open("./README.rst",mode="r")
long_description=description_file.read()

setup(
    #basic info
    name="hyperpose",
    version="2.1.1",
    #pack-up
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        "cython>=0.29",
        "numpy==1.16.4",
        "easydict>=1.9,<=1.10",
        "opencv-python>=3.4,<3.5",
        "tensorflow==2.3.1",
        "tensorlayer==2.2.3",
        "pycocotools"
    ],
    #meta data
    author="Hyperpose community",
    author_email="1137743903@qq.com",
    description="HyperPose is a library for building human pose estimation systems that can efficiently operate in the wild.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 license",
    keywords="pose estimation platform",
    url="https://github.com/tensorlayer/hyperpose",
    project_url={
        "Source Code":"https://github.com/tensorlayer/hyperpose",
        "Documentation":"https://hyperpose.readthedocs.io/en/latest/"
    }
)