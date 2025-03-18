import matplotlib.pyplot as plt

def sub_plots(lists):
    length = len(lists)
    if 1 <= length <=3:
        rows = 1
        cols = length
        position = range(1, length+1)
        
        fig = plt.figure(figsize = (9*cols,9))
        fig.tight_layout(pad = 0)       

    elif length == 4:
        rows = 2
        cols = 2
        position = range(1, length+1)
        
        fig = plt.figure(figsize = (9*cols,9))
        fig.tight_layout(pad = 0)
        
    else: # length >= 5
        cols = 3
        if length%cols == 0:
            rows = (length//cols)
        else:
            rows = (length//cols)+1
        position = range(1, length+1)
        
        fig = plt.figure(figsize = (9*cols,9))
        fig.tight_layout(pad = 0)
        
    return fig, rows, cols, length, position 
