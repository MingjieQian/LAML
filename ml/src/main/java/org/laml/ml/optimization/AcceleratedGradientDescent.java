package org.laml.ml.optimization;

import static org.laml.la.utils.InPlaceOperator.affine;
import static org.laml.la.utils.InPlaceOperator.assign;
import static org.laml.la.utils.Matlab.eps;
import static org.laml.la.utils.Matlab.eye;
import static org.laml.la.utils.Matlab.innerProduct;
import static org.laml.la.utils.Matlab.isnan;
import static org.laml.la.utils.Matlab.minus;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.ones;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rand;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.setMatrix;
import static org.laml.la.utils.Matlab.sum;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Printer.display;
import static org.laml.la.utils.Printer.fprintf;

import java.util.ArrayList;

import org.laml.la.matrix.Matrix;

/**
 * A Java implementation for the accelerated gradient descent method.
 * It is a general algorithm interface, only gradient and objective
 * function value are needed to compute outside the class.
 * </p>
 * A simple example: </br></br>
 * <code>
 * double epsilon = ...; // Convergence tolerance</br>
 * Matrix W = ...; // Initial matrix (vector) you want to optimize</br>
 * Matrix G = ...; // Gradient at the initial matrix (vector) you want to optimize</br>
 * double fval = ...; // Initial objective function value</br>
 * </br>
 * boolean flags[] = null; </br>
 * while (true) { </br>
 * &nbsp flags = AcceleratedGradientDescent.run(G, fval, epsilon, W); // Update W in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if it converges</br>
 * &nbsp &nbsp break; </br>
 * &nbsp fval = ...; // Compute the new objective function value at the updated W</br>
 * &nbsp if (flags[1])  // flags[1] indicates if gradient at the updated W is required</br>
 * &nbsp &nbsp G = ...; // Compute the gradient at the new W</br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Jan. 26th, 2014
 * 
 * @author Mingjie Qian
 */
public class AcceleratedGradientDescent {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int n = 10;
		Matrix t = rand(n);
		Matrix C = minus(t.mtimes(t.transpose()), times(0.1, eye(n)));
		Matrix y = times(3, minus(0.5, rand(n, 1)));
		double epsilon = 1e-4;
		double gamma = 0.01;
		
		// AcceleratedProximalGradient.prox = new ProxPlus();
		
		long start = System.currentTimeMillis();
		
		/*
		 *      min_x || C * x - y ||_2 + gamma * || x ||_2
	     * s.t. x
	     * 
	     * g(x) = || C * x - y ||_2 + gamma * || x ||_2
	     * h(x) = 0
		 */
		Matrix x0 = rdivide(ones(n, 1), n);
		Matrix x = x0.copy();
		
		Matrix r_x = null;
		double f_x = 0;
		double phi_x = 0;
		double gval = 0;
		double hval = 0;
		double fval = 0;
		
		r_x = C.mtimes(x).minus(y);
		f_x = norm(r_x);
		phi_x = norm(x);
		gval = f_x + gamma * phi_x;
		hval = 0;
		fval = gval + hval;
		
		Matrix Grad_f_x = null;
		Matrix Grad_phi_x = null;
		Matrix Grad = null;
		
		Grad_f_x = rdivide(C.transpose().mtimes(r_x), f_x);
		Grad_phi_x = rdivide(x, phi_x);
		Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
		
		boolean flags[] = null;
		int k = 0;
		int maxIter = 10000;
		hval = 0;
		while (true) {
			
			// flags = AcceleratedProximalGradient.run(Grad, gval, hval, epsilon, x);
			flags = AcceleratedGradientDescent.run(Grad, fval, epsilon, x);
			
			if (flags[0])
				break;
			
			if (sum(sum(isnan(x))) > 0) {
				int a = 1;
				a = a + 1;
			}
			
			/*
			 *  Compute the objective function value, if flags[1] is true
			 *  gradient will also be computed.
			 */
			r_x = C.mtimes(x).minus(y);
			f_x = norm(r_x);
			phi_x = norm(x);
			gval = f_x + gamma * phi_x;
			hval = 0;
			fval = gval + hval;
			
			if (flags[1]) {
				
				k = k + 1;
				
				// Compute the gradient
				if (k > maxIter)
					break;
				
				Grad_f_x = rdivide(C.transpose().mtimes(r_x), f_x);
				if (phi_x != 0)
					Grad_phi_x = rdivide(x, phi_x);
				else
					Grad_phi_x = times(0, Grad_phi_x);
				Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
				
			}
			
		}
		
		Matrix x_accelerated_proximal_gradient = x;
		double f_accelerated_proximal_gradient = fval;
		fprintf("fval_accelerated_proximal_gradient: %g\n\n", f_accelerated_proximal_gradient);
		fprintf("x_accelerated_proximal_gradient:\n");
		display(x_accelerated_proximal_gradient.transpose());
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n", elapsedTime);

	}
	
	/**
	 * Proximity operator: prox_th(X).
	 */
	private static ProximalMapping prox = new Prox();
	
	/**
	 * Current gradient.
	 */
	private static Matrix Grad_Y_k = null;
	
	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static Matrix X = null;
	
	/**
	 * Last matrix variable that we want to optimize.
	 */
	private static Matrix X_pre = null;
	
	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static Matrix Y = null;
	
	/**
	 * X_{k + 1} = prox_th(Y_k - t * Grad_Y_k) = y_k - t * G_Y_k.
	 */
	private static Matrix G_Y_k = null;
	
	/**
	 * g(Y_k).
	 */
	private static double gval_Y_k = 0;
	
	/**
	 * h(Y_k).
	 */
	private static double hval_Y_k = 0;
	
	/**
	 * f(Y_k) = g(Y_k) + h(Y_k).
	 */
	private static double fval_Y_k = 0;
	
	/**
	 * If gradient is required for the next step.
	 */
	private static boolean gradientRequired = false;
	
	/**
	 * If the algorithm converges or not.
	 */
	private static boolean converge = false;
	
	/**
	 * State for the automata machine.
	 * 0: Initialization
	 * 1: Before backtracking line search
	 * 2: Backtracking line search
	 * 3: After backtracking line search
	 * 4: Convergence
	 */
	private static int state = 0;
	
	/**
	 * Step length for backtracking line search.
	 */
	private static double t = 1;
	
	private static double beta = 0.95;
	
	/**
	 * Iteration counter.
	 */
	private static int k = 1;
	
	private static double xi = 1;
	
	/**
	 * 0: y_{k+1} = x_{k + 1} + (k / (k + 3)) * (x_{k + 1} - x_k)
	 * 1: y_{k+1} = x_{k + 1} + u * (x_{k + 1} - x_k),
	 *    u := 2 * (xi_{k + 1} - 1) / (1 + sqrt(1 + 4 * xi_{k + 1}^2))
	 */
	private static int type = 0;
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the accelerated proximal gradient algorithm. 
	 * The matrix variable to be optimized will be updated in place 
	 * to a better solution point with lower objective function value.
	 * 
	 * @param Grad_t gradient at X_t, required on the first revocation
	 * 
	 * @param gval_t g(X_t)
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param X_t current matrix variable to be optimized, will be
	 *            updated in place to a better solution point with
	 *            lower objective function value
	 *
	 * @return a {@code boolean} array with two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(Matrix Grad_t, double gval_t, double epsilon, Matrix X_t) {
		
		// If the algorithm has converged, we do a new job
		if (state == 4) {
			J.clear();
			X_pre = null;
			t = 1;
			k = 1;
			state = 0;
		}
		
		if (state == 0) {

			X = X_t.copy();
			Y = X_t.copy();
			
			gval_Y_k = gval_t;
			hval_Y_k = 0;
			fval_Y_k = gval_Y_k + hval_Y_k;
			if (Double.isNaN(fval_Y_k)) {
				System.err.println("Object function value is nan!");
				System.exit(1);
			}
			System.out.format("Initial ofv: %g\n", fval_Y_k);

			k = 1;
			xi = 1;
			t = 1;
			state = 1;

		}
		
		if (state == 1) {
			
			if (Grad_t == null) {
				System.err.println("Gradient is required!");
				System.exit(1);
			}
			
			Grad_Y_k = Grad_t.copy();
			/*if (Grad_Y_k == null)
				Grad_Y_k = Grad_t.copy();
			else
				assign(Grad_Y_k, Grad_t);*/
			
			gval_Y_k = gval_t;
			hval_Y_k = 0;
			
			
			state = 2;
			
			// X_t.setSubMatrix(plus(X, times(t, p)).getData(), 0, 0);
			// setMatrix(X_t, prox.compute(t, minus(Y, times(t, Grad_Y_k))));
			prox.compute(X_t, t, minus(Y, times(t, Grad_Y_k)));
			
			G_Y_k = rdivide(minus(Y, X_t), t);
			/*if (G_Y_k == null)
				G_Y_k = rdivide(minus(Y, X_t), t);
			else
				affine(G_Y_k, 1 / t, Y, -1 / t, X_t);*/
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {

			converge = false;

			if (gval_t <= gval_Y_k - t * innerProduct(Grad_Y_k, G_Y_k) + t / 2 * innerProduct(G_Y_k, G_Y_k) + eps) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				gradientRequired = false;
				setMatrix(X_t, prox.compute(t, minus(Y, times(t, Grad_Y_k))));
				G_Y_k = rdivide(minus(Y, X_t), t);
				return new boolean[] {converge, gradientRequired};
			}

		}
		
		if (state == 3) {
			
			double norm_G_Y = norm(G_Y_k);
			
			if (norm_G_Y < epsilon) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("Accelerated gradient descent method " +
						"converges with norm(G_Y_k) %f\n", norm_G_Y);
				return new boolean[] {converge, gradientRequired};
			}
			
			fval_Y_k = gval_Y_k + hval_Y_k;
		    J.add(fval_Y_k);
		    System.out.format("Iter %d, ofv: %g, norm(G_Y_k): %g\n", k, fval_Y_k, norm(G_Y_k));
			
		    // X_pre = X.copy();
		    if (X_pre == null)
				X_pre = X.copy();
			else
				assign(X_pre, X);
		    
			// X = X_t.copy();
		    assign(X, X_t);
		    
		    if (type == 0) {
		    	double s = (double)(k) / (k + 3);
		    	// Y = plus(X, times((double)(k) / (k + 3), minus(X, X_pre)));
		    	affine(Y, 1 + s, X, -s, X_pre);
		    } else if (type == 1) {
		    	double u = 2 * (xi - 1) / (1 + Math.sqrt(1 + 4 * xi * xi));
		    	affine(Y, 1 + u, X, -u, X_pre);
		    	xi = (1 + Math.sqrt(1 + 4 * xi * xi)) / 2;
		    }

			// setMatrix(X_t, Y);
			assign(X_t, Y);
			
			k = k + 1;
		    
		    state = 1;
			
		}
		
		converge = false;
	    gradientRequired = true;
	    return new boolean[] {converge, gradientRequired};
		
	}
	
}
