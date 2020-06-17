from setuptools import setup, find_packages

setup(name='pytorch-esn',
      version='1.2.4',
      packages=find_packages(),
      install_requires=[
          'torch',
          'torchvision',
          'numpy',
          'click'
      ],
      include_package_data=True,
      entry_points='''
        [console_scripts]
        fn_mackey_glass=torchesn.cmdline.fn_mackey_glass:executeESN
        fn_autotune=torchesn.cmdline.fn_autotune:tuneESN
        fn_cotrend=torchesn.cmdline.fn_cotrend:executeESN
      ''',
      description="Echo State Network module for PyTorch.",
      author='Stefano Nardo',
      author_email='stefano_nardo@msn.com',
      license='MIT',
      url="https://github.com/stefanonardo/pytorch-esn"
      )
