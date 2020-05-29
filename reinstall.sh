# python setup.py install --record files.txt
# xargs rm -rf < files.txt

python setup.py install --user --record files.txt; xargs rm -rf < files.txt; python setup.py install --user --record files.txt
