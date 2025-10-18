import numpy as np, cpp_hf
nk,d=4,2
w=np.ones((nk,nk))*((2/nk)*(2/nk)/(2*np.pi)**2)
H=np.zeros((nk,nk,d,d),np.complex128)
K=np.linspace(-1,1,nk)
V=(1.0/np.sqrt((K[:,None]**2+K[None,:]**2)+0.2)).astype(np.complex128)[...,None,None]
P0=np.zeros_like(H); ne=0.5*d*w.sum()
P,F,E,mu,n=cpp_hf.hartreefock_iteration_cpp(w,H,V,P0,ne,0.2,1,1e-2,2,1.0)
print("wheel ok", int(n), float(mu))