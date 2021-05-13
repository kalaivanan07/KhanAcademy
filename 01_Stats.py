# khan academy stats
import matplotlib.pyplot as plt  # To visualize
# generate related variables
import random
import numpy as np 
from numpy import arange
from numpy import mean
from numpy import std
from numpy import cov
from numpy import cumsum
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import norm 
from scipy import integrate
import math 
from math import erf
from math import sqrt 
import pandas as pd
from ast import literal_eval
import datetime 

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
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()


    # perfect negative linear correlation 
    data1 = 20 * randn(1000) + 100
    data2 = -1  * data1 
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()


    # positve strong linear correlation 
    data1 = 20 * randn(1000) + 100
    data2 = data1 + (10 * randn(1000) + 50)
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()

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
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()

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
    print(p_corr[0])
    print(s_corr)
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    plt.scatter(data1, data2)
    plt.show()

def plot_two_pdf(x1_axis, x2_axis):
    
    x1_axis.sort()
    x2_axis.sort()
    plt.plot(x1_axis, my_pdf(x1_axis)[0], 'grey')
    plt.plot(x2_axis, my_pdf(x2_axis)[0], 'blue')
    return 'Hi'

def sigmoid():
    xl = np.linspace(-10, 10, 100)
    #x =np.array([0]*len(xl))
    #yl = np.linspace(-100, 100, 100)
    ey = 1/ (1 + np.exp(xl))
    plot_graph(xl, ey, 'red')
    #plt.plot(xl, -xl, color = 'yellow')

def nonconvex():
    x = np.linspace(-2, 2, 100)
    y = x**4 + x**3 -2*x**2 -2*x
    plot_graph(x,y, 'grey')
    
def convex():
    x = np.linspace(-2, 2, 100)
    y = x**2
    plot_graph(x,y, 'blue')

def sci_pdf(x_axis):
    # plotting a normal distribution using python lib
    #x_axis = 20 * randn(100) + 50
    x_axis.sort()
    mu   = mean(x_axis)
    si   = std(x_axis)
    ndis = norm.pdf(x_axis, mean(x_axis), std(x_axis))
    plot_graph(range(len(x_axis)), ndis, 'grey')
    #print(ndis)
    return (ndis, mu, si)

def sci_cdf(x_axis, a, b):
    # plotting cdf using python lib
    #x_axis = 20 * randn(100) + 50
    x_axis.sort()
    ndis = norm.pdf(x_axis, mean(x_axis), std(x_axis))
    cdf = cumsum(ndis)
    print(cdf)
    plot_graph(range(len(x_axis)), cdf, 'grey')   

def i_pdf(x_axis, a, b):
    # plotting a normal distribution using math formula
    x_axis.sort()
    y_ndis = []
    mu   = mean(x_axis)
    si   = std(x_axis)
    print(si, mu)
    for eac in x_axis:
        z = (eac-mu)/si
        y_ndis.append(1/(math.sqrt( 2 * math.pi * si**2 * math.e**(z**2))))
    plot_graph(x_axis, y_ndis, 'grey')
    fill_area(x_axis,y_ndis,si,mu,a,b)
    area_intgrl(a, b)
    # return (y_ndis, mu, si)
    return 

def i_cdf(x_axis):
    # plotting cumulative distributive function 
    y_ndis = i_pdf(x_axis)
    y_cdf = []
    y_sum = 0
    for eac in range(len(y_ndis)):
        y_sum += y_ndis[eac]
        y_cdf.append(y_sum)
    plot_graph(range(len(x_axis)), y_cdf, 'grey')
    return y_cdf

def plot_graph(x_axis, y_axis, color):
    plt.figure(figsize=(10,8))
    plt.plot(x_axis , y_axis, color = color)

def fill_area(x_axis, y_axis, si, mu, a, b):
    x_fill = []
    y_fill = []
    for i in range(x_axis.shape[0]):
        if x_axis[i] >= a and x_axis[i] <= b:
            x_fill.append(x_axis[i])
            y_fill.append(y_axis[i])
    #print(x_axis)
    #print(y_axis)
    plt.fill_between(x_fill, y_fill, color='#0b559f', alpha=1.0)
                     
def area_intgrl(a, b):
    '''
    gfg = lambda x: x**2 + x + 1
    # using scipy.integrate.quad() method
    geek = integrate.quad(gfg, 1, 4)
    print(geek)
    '''
    #(res, err) = area_curve(y_ndis, a, b)
    gfg = lambda x: 1/(math.sqrt( 2 * math.pi * si**2 * math.e**(((x-mu)/si)**2)))
    res = integrate.quad(gfg , a, b)
    print('Area under curve is: %.3f'% (res[0]*100))

def area_under_curve(x_axis, a, b):
    (y_ndis, mu, si) =  i_pdf(x_axis)
    
    # probability from Z=0 to lower bound
    double_prob = math.erf((a-mu) / (si*sqrt(2)))
    p_lower = double_prob/2
    print(f'\n Lower Bound: {round(p_lower,4)}')
    
    # probability from Z=0 to upper bound
    double_prob = math.erf( (b-mu) / (si*sqrt(2)) )
    p_upper = double_prob/2
    print(f'\n Upper Bound: {round(p_upper,4)}')
    print('Area under curve :%.3f'% (p_upper-p_lower))


# need to understand 
def cdff():
    # Create some test data
    dx = 0.01
    #X  = np.arange(-2, 2, dx)
    X  = np.arange(-2, 2, dx)
    Y  = np.exp(-X ** 2)
    
    # Normalize the data to a proper PDF
    Y /= (dx * Y).sum()
    
    # Compute the CDF
    CY = np.cumsum(Y * dx) 
    
    # Plot both
    plt.plot(X, Y)
    plt.plot(X, CY, 'r--')
    
    show()

def corona_graph(file):
    try:

        df = pd.read_csv(file, header=None, sep='\n')
        print(df)
        df = df[0].str.strip(',').apply(literal_eval).apply(pd.Series)
        print(df)
        df[0] = df[0].agg(lambda x: pd.to_datetime('-'.join(map(str, x))))
        print(df)
        
        plt.figure(figsize=(10,6))
        plt.plot(df[0], df[1], color = 'grey', linestyle='dashed')
        plt.plot(df[0], df[2], color = 'blue', linestyle='dashed')
        plt.legend(['Daily Cases', 'Daily cure']) 
        plt.show()
        plt.plot(df[0], df[3], color = 'red', linestyle='dashed')
        plt.legend(['Death count'])
        plt.show()
        
    except Exception as e:
        print(e)
