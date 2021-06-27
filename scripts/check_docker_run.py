import os
import unittest
import re
from distutils.version import StrictVersion


class LinuxCheck(unittest.TestCase):
    def test_cuda_driver(self):
        p = os.popen('cat /proc/driver/nvidia/version')
        output = p.read()
        p.close()
        self.assertTrue('NVIDIA' in output.upper(), 'NVIDIA Driver not found. Please visit '
                                                    'https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index'
                                                    '.html '
                                                    '#driver-installation')
        if 'NVIDIA' in output.upper():
            version_code = None
            for line in output.splitlines():
                if 'NVIDIA' in line:
                    for item in line.split(' '):
                        m = re.search('[0-9][0-9][0-9].[0-9][0-9].[0-9][0-9]', item)
                        if m is not None:
                            version_code = m.group(0)
                            break
            self.assertNotEqual(version_code, None, 'NVIDIA version not found...')
            if version_code is not None:
                self.assertGreaterEqual(StrictVersion(version_code), StrictVersion('418.81.07'), 'Your CUDA driver is '
                                                                                                 'old. Please upgrade '
                                                                                                 'it to >= 418.81.07 '
                                                                                                 'according '
                                                                                                 'to '
                                                                                                 'https://docs.nvidia'
                                                                                                 '.com/cuda/cuda'
                                                                                                 '-installation-guide'
                                                                                                 '-linux/index.html'
                                                                                                 '#driver-installation')

    def test_docker_version(self):
        p = os.popen("docker version --format '{{.Client.Version}}'")
        version_code = p.read()
        return_code = p.close()
        self.assertEqual(return_code, None, 'docker command not found...')
        self.assertNotEqual(version_code, None, 'Docker version cannot be found...')
        self.assertGreaterEqual(StrictVersion(version_code), StrictVersion('19.03'), 'Your docker version is too old '
                                                                                     'to support "--gpus" flag... '
                                                                                     'Please install a newer version '
                                                                                     '(>= 19.03) via '
                                                                                     'https://docs.docker.com/engine'
                                                                                     '/install/')

    def test_nvidia_docker(self):
        return_code = os.system("docker run --rm --gpus all nvidia/cuda:10.0-base nvidia-smi")
        self.assertEqual(return_code, 0, 'Docker with CUDA functionality cannot run properly. Please visit '
                                         'https://docs.nvidia.com/datacenter/cloud-native/container-toolkit'
                                         '/install-guide.html#pre-requisites to install latest nvidia container')


if __name__ == '__main__':
    unittest.main(verbosity=2)
