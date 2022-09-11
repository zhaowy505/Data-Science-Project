# coding: utf-8
import sys
import os

DEBUG = True
os.environ['DEBUG'] = str(DEBUG)

app_name = 'Decision tree model visualization'
base_dir = os.path.abspath(os.path.dirname(__file__))
home_dir = os.path.expanduser('~')
app_config_fp = os.path.join(home_dir, 'qt.{}'.format(app_name))

log_file = os.path.join(base_dir, 'log.log')
image_width = 1024
image_height = 768

time_format = '%Y-%m-%d %H:%M:%S'

