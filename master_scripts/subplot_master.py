import matplotlib.pyplot as plt

def subplots(lists):
    length = len(lists)
    if length >= 1 && length <=3:
        rows == 1
        cols == length
        position = range(1, length+1)
        
        fig = plt.figure(figsize = (12,12))
        
        for i in range(length):
            ax = fig.add_subplot(rows, cols, position[i])
            ax.plot(x[i], y[i], marker = 'o', color = 'r')
        plt.show()

        
