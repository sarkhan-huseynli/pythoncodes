import numpy as nd

def gradientDescent(X,y, theta, alpha, num_iters):
	m=len(y)
	J_history = np.zeros(num_iters, 1);\
	
	for i in num_iters
	    predictions=X*theta;
    
   	    cost=predictions-y;
    
            theta=theta-(alpha*(1/m)*X'*cost);

	return theta
