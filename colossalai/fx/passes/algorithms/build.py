from distutils.core import setup, Extension

module1 = Extension('c_version_dp', sources=['dynamic_programs.c'])

setup(name='a demo c extension', version='0.1', description='This is a demo package', ext_modules=[module1])
