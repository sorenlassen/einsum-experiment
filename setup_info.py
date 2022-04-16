# -*- coding: utf-8 -*-
import os
import re
import ast
import configparser

def path(to):
  return os.path.join(os.path.dirname(__file__), to)

def ev(x):
  x = re.sub(r'^(.+?)\s*[#].*$', r'\1', x) # strip inline comments
  x = re.sub(r'[{]\s*([^\s=]+)\s*=\s*', r"{'\1': ", x) # convert toml dict to python dict
  return False if x == 'false' else True if x == 'true' else ast.literal_eval(x)

config = configparser.ConfigParser()
config.read(path('pyproject.toml'))
config = {k.lower(): {k1.lower(): ev(v1) for k1, v1 in v.items()} for k, v in config.items()}

info = config['tool.poetry']
package = info['name'].replace('-', '_')
author, author_email = re.findall(r"^(.+?)\s*(?:[<](.+?)[>])?$", info['authors'][0])[0]



base_kwargs = {
    'name': info['name'],
    'version': info['version'],
    'description': info.get('description'),
    'long_description': open(path(info.get('readme', 'README.md'))).read(),
    'author': author,
    'author_email': author_email,
    'url': info.get('homepage'),
    'license': info.get('license'),
}

# prune blank strings or None values
base_kwargs = {k: v for k, v in base_kwargs.items() if v}

# defaults

package_dir = \
{'': 'src'}

packages = \
[package]

package_data = \
{'': ['*']}

install_requires = \
[]

