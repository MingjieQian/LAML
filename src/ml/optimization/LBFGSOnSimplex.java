package ml.optimization;

import static la.io.IO.loadMatrix;
import static ml.utils.InPlaceOperator.affine;
import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.minus;
import static ml.utils.InPlaceOperator.minusAssign;
import static ml.utils.InPlaceOperator.plusAssign;
import static ml.utils.InPlaceOperator.subplusAssign;
import static ml.utils.Matlab.eye;
import static ml.utils.Matlab.gt;
import static ml.utils.Matlab.innerProduct;
import static ml.utils.Matlab.isnan;
import static ml.utils.Matlab.logicalIndexing;
import static ml.utils.Matlab.logicalIndexingAssignment;
import static ml.utils.Matlab.lt;
import static ml.utils.Matlab.max;
import static ml.utils.Matlab.minus;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.not;
import static ml.utils.Matlab.ones;
import static ml.utils.Matlab.or;
import static ml.utils.Matlab.plus;
import static ml.utils.Matlab.rand;
import static ml.utils.Matlab.rdivide;
import static ml.utils.Matlab.size;
import static ml.utils.Matlab.sum;
import static ml.utils.Matlab.times;
import static ml.utils.Matlab.uminus;
import static ml.utils.Printer.display;
import static ml.utils.Printer.fprintf;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import la.matrix.Matrix;
import la.matrix.SparseMatrix;


/**
 * A Java implementation for the projected limited-memory BFGS algorithm
 * with simplex constraints.
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
 * &nbsp flags = LBFGSOnSimplex.run(G, fval, epsilon, W); // Update W in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if L-BFGS converges</br>
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
public class LBFGSOnSimplex {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int n = 10;
		Matrix t = rand(n);
		Matrix C = minus(t.mtimes(t.transpose()), times(0.05, eye(n)));
		Matrix y = times(3, rand(n, 1));
		double epsilon = 1e-6;
		double gamma = 0.01;
		
		String path = "C:/Aaron/My Codes/Matlab/Convex Optimization";
		C = loadMatrix(path + File.separator + "C.txt");
		y = loadMatrix(path + File.separator + "y.txt");
		
		long start = System.currentTimeMillis();
		
		/*
		 *      min_x || C * x - y ||_2 + gamma * || x ||_2
	     * s.t. sum_i x_i = 1
         *      x_i >= 0
		 */
		Matrix x0 = rdivide(ones(n, 1), n);
		Matrix x = x0.copy();
		
		Matrix r_x = null;
		double f_x = 0;
		double phi_x = 0;
		double fval = 0;
		
		r_x = C.mtimes(x).minus(y);
		f_x = norm(r_x);
		phi_x = norm(x);
		fval = f_x + gamma * phi_x;
		
		Matrix Grad_f_x = null;
		Matrix Grad_phi_x = null;
		Matrix Grad = null;
		
		Grad_f_x = rdivide(C.transpose().mtimes(r_x), f_x);
		Grad_phi_x = rdivide(x, phi_x);
		Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
		
		boolean flags[] = null;
		int k = 0;
		int maxIter = 1000;
		while (true) {
			flags = LBFGSOnSimplex.run(Grad, fval, epsilon, x);
			if (flags[0])
				break;
			// display(W);
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
			fval = f_x + gamma * phi_x;
			
			
			if (flags[1]) {
				k = k + 1;
				// Compute the gradient
				if (k > maxIter)
					break;
				
				Grad_f_x = rdivide(C.transpose().mtimes(r_x), f_x);
				Grad_phi_x = rdivide(x, phi_x);
				Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
			}
			
		}
		
		Matrix x_projected_LBFGS_Armijo = x;
		double f_projected_LBFGS_Armijo = fval;
		fprintf("fval_projected_LBFGS_Armijo: %g\n\n", f_projected_LBFGS_Armijo);
		fprintf("x_projected_LBFGS_Armijo:\n");
		display(x_projected_LBFGS_Armijo.transpose());
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n", elapsedTime);
		
	}
	
	/**
	 * Current gradient.
	 */
	private static Matrix G = null;
	
	/**
	 * Current projected gradient.
	 *//*
	private static Matrix PG = null;*/
	
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
	 *//*
	private static Matrix p = null;*/
	
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
	
	private static Matrix z = null;
	
	private static Matrix z_t = null;
	
	private static Matrix p_z = null;
	
	private static Matrix I_z = null;
	
	private static Matrix G_z = null;
	
	private static Matrix PG_z = null;
	
	private static int i = -1;
	
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
			z_t = null;
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
			
			tol = epsilon * norm(G);
			
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
			
			/*I_k = or(lt(G, 0), gt(X, 0));
			I_k_com = not(I_k);
			PG = G.copy();
			// logicalIndexingAssignment(PG, I_k, logicalIndexing(G, I_k));
			logicalIndexingAssignment(PG, I_k_com, 0);*/
			
			/*double norm_PGrad = norm(PG);
			if (norm_PGrad < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
				return new boolean[] {converge, gradientRequired};
			}*/
		    
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
			
			// q = G;
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
			
			/*Matrix HG = r;
			Matrix PHG = HG.copy();
			I_k = or(lt(HG, 0), gt(X, 0));
			I_k_com = not(I_k);
			logicalIndexingAssignment(PHG, I_k_com, 0);
			if (innerProduct(PHG, G) <= 0)
				p = uminus(PG);
			else
				p = uminus(PHG);*/
			
			// Transform x_k to z_k
			// i = (int) max(X, 1).get("idx").getEntry(0, 0) + 1;
			i = (int) max(X, 1)[1].get(0);
			int n = size(X, 1);
			/*Matrix A = vertcat(horzcat(eye(i - 1), zeros(i - 1, n - i)),
					horzcat(uminus(ones(1, i - 1)), uminus(ones(1, n - i))),
					horzcat(zeros(n - i, i - 1), eye(n - i)));
			
			Matrix b_z = vertcat(zeros(i - 1, 1), ones(1, 1), zeros(n - i, 1));*/
			Matrix A = new SparseMatrix(n, n - 1);
			for (int j = 0; j < i; j++)
				A.setEntry(j, j, 1);
			for (int j = 0; j < n - 1; j++)
				A.setEntry(i, j, -1);
			for (int j = i; j < n - 1; j++)
				A.setEntry(j + 1, j, 1);
			
			Matrix b_z = new SparseMatrix(n, 1);
			b_z.setEntry(i, 0, 1);
			
			I_z = not(b_z);
			z = logicalIndexing(X, I_z);
			if (z_t == null)
				z_t = z.copy();
			
			G_z = A.transpose().mtimes(G);
			I_k = or(lt(G_z, 0), gt(z, 0));
			I_k_com = not(I_k);
			PG_z = G_z.copy();
			logicalIndexingAssignment(PG_z, I_k_com, 0);
			
			double norm_PGrad_z = norm(PG_z);
			if (norm_PGrad_z < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS on simplex converges with norm(PGrad_z) %f\n", norm_PGrad_z);
				return new boolean[] {converge, gradientRequired};
			}
			
			Matrix HG_z = A.transpose().mtimes(r);
			I_k = or(lt(HG_z, 0), gt(z, 0));
			I_k_com = not(I_k);
			Matrix PHG_z = HG_z.copy();
			logicalIndexingAssignment(PHG_z, I_k_com, 0);
			
			if (innerProduct(PHG_z, G_z) <= 0)
				p_z = uminus(PG_z);
			else
				p_z = uminus(PHG_z);
			
			t = 1;
			// z is always less than 0
			// z = innerProduct(G, p);
			// Matrix z_t = null;
			while (true) {
				// z_t = subplus(plus(z, times(t, p_z)));
				affine(z_t, t, p_z, '+', z);
				subplusAssign(z_t);
				if (sum(sum(z_t)) <= 1) {
					break;
				} else {
					t = beta * t;
				}
			}
			
			state = 2;
			
			// z_t = subplus(plus(z, times(t, p_z)));
			// setMatrix(X_t, X);
			logicalIndexingAssignment(X_t, I_z, z_t);
			// X_t.setEntry(i - 1, 0, 1 - sum(sum(z_t)));
			X_t.setEntry(i, 0, 1 - sum(sum(z_t)));
			
			// X_t.setSubMatrix(subplus(plus(X, times(t, p))).getData(), 0, 0);
			// setMatrix(X_t, subplus(plus(X, times(t, p))));
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {
			
			converge = false;

			if (fval_t <= fval + alpha * innerProduct(G_z, minus(z_t, z))) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				// z_t = subplus(plus(z, times(t, p_z)));
				affine(z_t, t, p_z, '+', z);
				subplusAssign(z_t);
				logicalIndexingAssignment(X_t, I_z, z_t);
				X_t.setEntry(i, 0, 1 - sum(sum(z_t)));
				gradientRequired = false;
				// setMatrix(X_t, subplus(plus(X, times(t, p))));
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
		    
		    /*if (Math.abs(fval_t - fval) < eps) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(PGrad_z): %g\n", k + 1, fval, norm(PG_z));
				return new boolean[] {converge, gradientRequired};
		    }*/
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(PGrad_z): %g\n", k + 1, fval, norm(PG_z));
		    
		    // X = X_t.copy();
		    assign(X, X_t);
		    // G = Grad_t.copy();
		    assign(G, Grad_t);
		    
		    /*s_k = X.minus(X_pre);
		    y_k = minus(G, G_pre);
		    rou_k = 1 / innerProduct(y_k, s_k);
		    
		    // Now s_ks, y_ks, and rou_ks all have k elements
		    if (k >= m) {
		    	s_ks.removeFirst();
		    	y_ks.removeFirst();
		    	rou_ks.removeFirst();
		    }*/
		 
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
