### For loop vs Numpy


```python
import numpy as np
a = np.array([1,2,3,4])
print(a)
```

    [1 2 3 4]
    


```python
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("Vectorized version: " + str(1000*(toc-tic)) + "ms")
```

    249858.62285772397
    Vectorized version: 3.0040740966796875ms
    


```python
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("For loop:" + str(1000*(toc-tic)) + "ms")
```

    249858.62285773095
    For loop:564.8174285888672ms
    

약 100배 이상 차이가 난다.
