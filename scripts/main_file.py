"""
This script run the replication of the project
"""
import os
import sys
sys.path.append(r"C:/Users/zach_/Desktop/Research/Github/RFE_project/")

print("1. making the data")
os.system("python -m Src.make_data" )
print("2. running the model")
os.system("python -m Src.model")


