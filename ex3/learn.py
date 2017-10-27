import matplotlib.pyplot as plt
import scipy as sp 

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

x = data[:,0]
y  = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x,y,s = 5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/Hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i' %w for w in range(10)])
plt.autoscale(tight = True)
plt.grid()


# ERROR FUNCTION
def error(f, x, y):
    return sp.sum((f(x)- y)**2)

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 4, full=True)

print ("Model parameters: %s" %fp1)

f1 = sp.poly1d(fp1)
print (error(f1, x, y))

fx = sp.linspace(0, x[-1], 1000) 
plt.plot(fx, f1(fx), linewidth=3, c='r',label="f1")


###########
inflection = int(3.5*7*24)
xa = x[:inflection]
xb = x[inflection:]

ya = y[:inflection]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa,ya,1))
fb = sp.poly1d(sp.polyfit(xb,yb,1))

plt.plot(fx, fa(fx), linewidth=2, c='g')
plt.plot(fx, fb(fx), linewidth=2, c='y')

plt.legend(["d=%i" %f1.order], loc="upper left")








plt.show()