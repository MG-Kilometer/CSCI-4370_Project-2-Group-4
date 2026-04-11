# -*- coding: utf-8 -*-
"""
Data Mining Project 2
Authors : Miles Glover, Madison Nicholson, Victory Orobosa
"""

from openpyxl import load_workbook

#Load Dataset
def load_XLSX(file_name):

    print("Loading file")

    dataset = []

    wb = load_workbook(file_name)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))

    header = list(rows[0])

    #construct dataset row-by-row
    for row in rows[1:]:
        dataset.append(list(row))

    print("File loaded\n")

    return dataset,header

#Preproccesing / Normalization
#functions

#Clustering Algorithm Implementation
#functions

# K-Means
#functions

# Fuzzy C-Means
#functions

#Evaluation
#functions

if __name__ == '__main__':
    
    
    #load data
    data,header = load_XLSX("Longotor1delta.xlsx")

    #data is structured so data[n], is the nth data row
    
    
    
    print("Program Finished")
    