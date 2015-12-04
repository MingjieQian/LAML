package ml.optimization;

import static ml.utils.InPlaceOperator.affine;
import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.minus;
import static ml.utils.InPlaceOperator.minusAssign;
import static ml.utils.InPlaceOperator.plusAssign;
import static ml.utils.InPlaceOperator.subplusAssign;
import static ml.utils.Matlab.eps;
import static ml.utils.Matlab.gt;
import static ml.utils.Matlab.inf;
import static ml.utils.Matlab.innerProduct;
import static ml.utils.Matlab.logicalIndexingAssignment;
import static ml.utils.Matlab.lt;
import static ml.utils.Matlab.minus;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.not;
import static ml.utils.Matlab.or;
import static ml.utils.Matlab.times;
import static ml.utils.Matlab.uminus;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import la.matrix.Matrix;

/**
 * A Java implementation for the projected limited-memory BFGS algorithm
 * with nonnegative constraints.
 * 
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
 * &nbsp flags = NonnegativePLBFGS.run(G, fval, epsilon, W); // Update W in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if L-BFGS converges</br>
 * &nbsp &nbsp break; </br>
 * &nbsp fval = ...; // Compute the new objective function value at the updated W</br>
 * &nbsp if (flags[1])  // flags[1] indicates if gradient at the updated W is required</br>
 * &nbsp &nbsp G = ...; // Compute the gradient at the new W</br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Jan. 24th, 2014
 * 
 * @author Mingjie Qian
 */
public class NonnegativePLBFGS {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	/**
	 * Current gradient.
	 */
	private static Matrix G = null;
	
	/**
	 * Current projected gradient.
	 */
	private static Matrix PG = null;
	
	/**
	 * Last gradient.
	 */
	private static Matrix G_pre = null;
	
	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static Matrix X = null;
	
	/**
	 * Last matrix variable that we want to optimize.
	 */
	private static Matrix X_pre = null;
	
	/**
	 * Decreasing step.
	 */
	private static Matrix p = null;
	
	/**
	 * The last objective function value.
	 */
	private static double fval = 0;
	
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
	
	/**
	 * A temporary variable holding the inner product of the decreasing step p
	 * and the gradient G, it should be always non-positive.
	 *//*
	private static double z = 0;*/
	
	/**
	 * Iteration counter.
	 */
	private static int k = 0;
	
	private static double alpha = 0.2;
	
	private static double beta = 0.75;
	
	private static int m = 30;
	
	private static double H = 0;
		
	private static Matrix s_k = null;
	private static Matrix y_k = null;
	private static double rou_k;
	
	private static LinkedList<Matrix> s_ks = new LinkedList<Matrix>();
	private static LinkedList<Matrix> y_ks = new LinkedList<Matrix>();
	private static LinkedList<Double> rou_ks = new LinkedList<Double>();
	
	/**
	 * Tolerance of convergence.
	 */
	private static double tol = 1;
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the LBFGS algorithm. The matrix variable to
	 * be optimized will be updated in place to a better solution point with
	 * lower objective function value.
	 * 
	 * @param Grad_t gradient at original X_t
	 * 
	 * @param fval_t objective function value on original X_t
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param X_t current matrix variable to be optimized, will be
	 *            updated in place to a better solution point with
	 *            lower objective function value.
	 * 
	 * @return a {@code boolean} array of two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(Matrix Grad_t, double fval_t, double epsilon, Matrix X_t) {
		
		// If the algorithm has converged, we do a new job
		if (state == 4) {
			s_ks.clear();
			y_ks.clear();
			rou_ks.clear();
			J.clear();
			X_pre = null;
			G_pre = null;
			PG = null;
			state = 0;
		}
		
		if (state == 0) {
			
			X = X_t.copy();
			if (Grad_t == null) {
				System.err.println("Gradient is required on the first call!");
				System.exit(1);
			}
			G = Grad_t.copy();
			fval = fval_t;
			if (Double.isNaN(fval)) {
				System.err.println("Object function value is nan!");
				System.exit(1);
			}
			System.out.format("Initial ofv: %g\n", fval);
			
			tol = epsilon * norm(G, inf);
			
			k = 0;
			state = 1;
			
		}
		
		if (state == 1) {
						
			Matrix I_k = null;
			Matrix I_k_com = null;
			// Matrix PG = null;
			
			/*I_k = G_x < 0 | x > 0;
		    I_k_com = not(I_k);
		    PG_x(I_k) = G_x(I_k);
		    PG_x(I_k_com) = 0;*/
			
			I_k = or(lt(G, 0), gt(X, 0));
			I_k_com = not(I_k);
			// PG = G.copy();
			if (PG == null)
				PG = G.copy();
			else
				assign(PG, G);
			logicalIndexingAssignment(PG, I_k_com, 0);
			
			/*disp("PG:");
			disp(PG);*/
			
			double norm_PGrad = norm(PG, inf);
			if (norm_PGrad < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
				return new boolean[] {converge, gradientRequired};
			}	
		    
			if (k == 0) {
				H = 1;
			} else {
				H = innerProduct(s_k, y_k) / innerProduct(y_k, y_k);
			}	
			
			Matrix s_k_i = null;
			Matrix y_k_i = null;
			Double rou_k_i = null;
			
			Iterator<Matrix> iter_s_ks = null;
			Iterator<Matrix> iter_y_ks = null;
			Iterator<Double> iter_rou_ks = null;
			
			double[] a = new double[m];
			double b = 0;
			
			Matrix q = null;
			Matrix r = null;
			
			q = G.copy();
			iter_s_ks = s_ks.descendingIterator();
			iter_y_ks = y_ks.descendingIterator();
			iter_rou_ks = rou_ks.descendingIterator();
			for (int i = s_ks.size() - 1; i >= 0; i--) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				a[i] = rou_k_i * innerProduct(s_k_i, q);
				// q = q.minus(times(a[i], y_k_i));
				minusAssign(q, a[i], y_k_i);
			}
			r = times(H, q);
			iter_s_ks = s_ks.iterator();
			iter_y_ks = y_ks.iterator();
			iter_rou_ks = rou_ks.iterator();
			for (int i = 0; i < s_ks.size(); i++) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				b = rou_k_i * innerProduct(y_k_i, r);
				// r = r.plus(times(a[i] - b, s_k_i));
				plusAssign(r, a[i] - b, s_k_i);
			}
			// p is a decreasing step
			// p = uminus(r);
			
			/*HG_x = r;
		    I_k = HG_x < 0 | x > 0;
		    I_k_com = not(I_k);
		    PHG_x(I_k) = HG_x(I_k);
		    PHG_x(I_k_com) = 0;
		    
		    if (PHG_x' * G_x <= 0)
		        p = -PG_x;
		    else
		        p = -PHG_x;
		    end*/
			
			Matrix HG = r;
			Matrix PHG = HG.copy();
			I_k = or(lt(HG, 0), gt(X, 0));
			I_k_com = not(I_k);
			logicalIndexingAssignment(PHG, I_k_com, 0);
			if (innerProduct(PHG, G) <= 0)
				p = uminus(PG);
			else
				p = uminus(PHG);
			
			/*disp("p:");
			disp(p);*/
			
			t = 1;
			// z is always less than 0
			// z = innerProduct(G, p);
			
			state = 2;
			
			// setMatrix(X_t, subplus(plus(X, times(t, p))));
			affine(X_t, X, t, p);
			subplusAssign(X_t);
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {
			
			converge = false;

			if (fval_t <= fval + alpha * innerProduct(G, minus(X_t, X))) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				gradientRequired = false;
				// setMatrix(X_t, subplus(plus(X, times(t, p))));
				affine(X_t, X, t, p);
				subplusAssign(X_t);
			}	

			return new boolean[] {converge, gradientRequired};
			
		}
		
		if (state == 3) {
			
			// X_pre = X.copy();
			if (X_pre == null)
				X_pre = X.copy();
			else
				assign(X_pre, X);
		    
		    // G_pre = G.copy();
			if (G_pre == null)
				G_pre = G.copy();
			else
				assign(G_pre, G);
			/*disp("G_pre:");
		    disp(G_pre);*/
		    
		    if (Math.abs(fval_t - fval) < eps) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
				return new boolean[] {converge, gradientRequired};
		    }
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
		    
		    // X = X_t.copy();
		    assign(X, X_t);
		    // G = Grad_t.copy();
		    assign(G, Grad_t);
		    /*disp("Grad_t:");
		    disp(Grad_t);*/
		    
		    // Now s_ks, y_ks, and rou_ks all have k elements
		    if (k >= m) {
		    	s_k = s_ks.removeFirst();
		    	y_k = y_ks.removeFirst();
		    	rou_ks.removeFirst();
		    	minus(s_k, X, X_pre);
		    	minus(y_k, G, G_pre);
		    } else { // if (k < m)
		    	s_k = X.minus(X_pre);
		    	y_k = G.minus(G_pre);
		    }
		    
		    rou_k = 1 / innerProduct(y_k, s_k);
		    
		    s_ks.add(s_k);
	    	y_ks.add(y_k);
	    	rou_ks.add(rou_k);
		    
		    k = k + 1;
		    
		    state = 1;
		    
		}
		
		converge = false;
	    gradientRequired = false;
	    return new boolean[] {converge, gradientRequired};
		
	}

}
