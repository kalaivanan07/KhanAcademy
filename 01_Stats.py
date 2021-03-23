# khan academy stats

import matplotlib.pyplot as plt  # To visualize

# generate related variables
import random
from numpy import mean
from numpy import std
from numpy import cov
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def bar_plot(x, y):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x,y)
    plt.show()

# random number head tail
def prob_ht(n):
    # to show, at very large numbers the probability converges to equal half
    t = 0
    h = 0

    for i in range(n):
        if random.randint(0,1) == 0:
            h = h+1
        else:
            t = t+1

    print ('h=%i t=%i p=%.3f' %(h, t, (h/t)))
    bar_plot(['Head','Tail'], [h, t])

def prob_3roll(n):
    # throwing a dice n times and observing the probability 
    # of winning if the sum is greater than 10 in three counts
    # or lose if the sum is lesser than or equal to 10 in three counts
    
    count = 0 
    sum   = 0
    l     = 0
    w     = 0
    for i in range(n):
        r = random.randint(1,10)
        '''
        print('r' + str(r))
        print(l)
        print(w)
        '''
        if r <= 6:
            count += 1 
            sum = sum + r 
        if count == 3 and sum <= 10:
            l = l +1
            count = 0
            sum   = 0 
        elif count == 3 and sum > 10:    
            w = w +1
            count = 0
            sum   = 0 
    
    print('w=%i l=%i p=%.3f' % (w,l,(w/l)))
    bar_plot(['Win','Lose'], [w,l])

def std_dv():
    #relation between randomnly generated numbers
    # seed random number generator
    seed(1)
    # prepare data
    
    # perfect positive linear correlation 
    data1 = 20 * randn(1000) + 100
    data2 = 1  * data1 

    # perfect negative linear correlation 
    data1 = 20 * randn(1000) + 100
    data2 = 1  * data1 

    # positve strong linear correlation 
    data1 = 20 * randn(1000) + 100
    data2 = data1 + (10 * randn(1000) + 50)

    # negative strong linear correlation    
    data1 = 20 * randn(1000) + 100
    data2 = data1 + (10 * randn(1000) + 50)
    data2 = data2 * -1 
    
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()

def covr():
    seed(1)
    data1 = 25 *  randn(100) + 100 
    data2 = 50 *  randn(100) + 1000

    # covariance between two variables
    cvr = sum((data1 - mean(data1))*(data2 - mean(data2)))/(len(data1) - 1)
    print(cvr)
    
    # covariance (variance ) of a variable X with itself
    cvr = sum((data1 - mean(data1))*(data1 - mean(data1)))/(len(data1) - 1)
    print(cvr)
    
    # covariance (variance ) of a variable Y with itself
    cvr = sum((data2 - mean(data2))*(data2 - mean(data2)))/(len(data2) - 1)
    print(cvr)
    
    cvr = cov(data1, data2)
    print(cvr)

def corr():
    # finding coefficient correlation 
        
    seed(1)
    data1 = 25 *  randn(100) + 100
    data2 = 50 *  randn(100) + 1000
    
    # mathematical formula 
    corr = sum(((data1) - mean(data1))*((data2) - mean(data2))/(std(data1)*std(data2)))/(len(data1) - 1)
    
    # pearson method
    p_corr = pearsonr(data1, data2)
    
    # spearman 
    s_corr = spearmanr(data1, data2)
    print('corr=%.3f ' % corr )
    print(p_corr)
    print(s_corr)

