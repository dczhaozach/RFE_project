"""
This script run the replication of the project
"""
import os
import sys
sys.path.append(r"C:/Users/zach_/Desktop/Research/Github/RFE_project/")

os.system("python -m src.make_data" )
os.system("python -m src.model" )


