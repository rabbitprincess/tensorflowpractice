import numpy as np
import matplotlib.pyplot as plt
 
x = np.arange(0, 100)
y = x * 2
z = x ** 2
plt.subplot(    211   )
plt.plot(x,y,color =   "yellow"  ,   lw=  5)
plt.xlabel("X")
plt.ylabel("Y")
plt. title    ("First")
plt.subplot(   212    )
plt.plot(x,z,color="red",  lw   =3,  ls    ="--")
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Second")