from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer   

def func0(a):                                
    for i in range(10000000):
        a[i]+= 1     
        

@jit(target_backend='cpu')                         
def func1(a):
    for i in range(10000000):
        a[i]+= 1
        

@jit(target_backend='cuda')                         
def func2(a):
    for i in range(10000000):
        a[i]+= 1
        
        
if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
      
    start = timer()
    func0(a)
    print("with CPU normal:", timer()-start)    
        
    start = timer()
    func1(a)
    print("with CPU optimized:", timer()-start)   
      
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)
    
    
#with CPU normal: 1.419965000000957
#with CPU optimized: 0.046378500002902
#with GPU: 0.04513060000317637