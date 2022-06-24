# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:30:39 2022

@author: rh43233
"""

#import streamlit as st


#import os

import streamlit as st
#import pandas as pd
#import pickle

from PIL import Image
image = Image.open('w.jpg')

#import sys
#from streamlit import cli as stcli

#if __name__ == '__main__':
#    sys.argv = ["streamlit", "run", "APP_NAME.py"]
#    sys.exit(stcli.main())

st.title('welcome to Westfield')
st.image(image, caption='West')
