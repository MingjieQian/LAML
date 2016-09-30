package org.laml.ml.optimization;

import static org.laml.la.io.IO.loadMatrix;
import static org.laml.la.io.IO.saveMatrix;
import static org.laml.la.utils.InPlaceOperator.affine;
import static org.laml.la.utils.InPlaceOperator.assign;
import static org.laml.la.utils.Matlab.diag;
import static org.laml.la.utils.Matlab.eye;
import static org.laml.la.utils.Matlab.getRows;
import static org.laml.la.utils.Matlab.horzcat;
import static org.laml.la.utils.Matlab.innerProduct;
import static org.laml.la.utils.Matlab.lt;
import static org.laml.la.utils.Matlab.mldivide;
import static org.laml.la.utils.Matlab.mtimes;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.ones;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rand;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.size;
import static org.laml.la.utils.Matlab.sumAll;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Matlab.uminus;
import static org.laml.la.utils.Matlab.vertcat;
import static org.laml.la.utils.Matlab.zeros;
import static org.laml.la.utils.Printer.disp;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Time.pause;
import static org.laml.la.utils.Time.tic;
import static org.laml.la.utils.Time.toc;
import org.laml.la.matrix.Matrix;

/**
 * General quadratic programming:
 * <p>
 *      min 2 \ x' * Q * x + c' * x </br>
 * s.t. A * x = b </br>
 *      B * x <= d </br>
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 26th, 2014
 */
public class GeneralQP {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		/*
		 * Number of unknown variables
		 */
		int n = 5;
		
		/*
		 * Number of inequality constraints
		 */
		int m = 6;
		
		/*
		 * Number of equality constraints
		 */
		int p = 3;

		/*Matrix x = rand(n, n);
		Matrix Q = x.mtimes(x.transpose()).plus(times(rand(1), eye(n)));
		Matrix c = rand(n, 1);

		double HasEquality = 1;
		Matrix A = times(HasEquality, rand(p, n));
		x = rand(n, 1);
		Matrix b = A.mtimes(x);
		Matrix B = rand(m, n);
		double rou = -2;
		Matrix d = plus(B.mtimes(x), times(rou, ones(m, 1)));*/
		
		Matrix x = null;
		Matrix Q = null;
		Matrix c = null;
		Matrix A = null;
		Matrix b = null;
		Matrix B = null;
		Matrix d = null;
		double rou = -2;
		double HasEquality = 1;
		
		boolean generate = false;
		if (generate) {
			x = rand(n, n);
			Q = x.mtimes(x.transpose()).plus(times(rand(1), eye(n)));
			c = rand(n, 1);

			A = times(HasEquality, rand(p, n));
			x = rand(n, 1);
			b = A.mtimes(x);
			B = rand(m, n);
			d = plus(B.mtimes(x), times(rou, ones(m, 1)));

			saveMatrix("Q", Q);
			saveMatrix("c", c);
			saveMatrix("A", A);
			saveMatrix("b2", b);
			saveMatrix("B", B);
			saveMatrix("d", d);
		} else {
			Q = loadMatrix("Q");
			c = loadMatrix("c");
			A = loadMatrix("A");
			b = loadMatrix("b2");
			B = loadMatrix("B");
			d = loadMatrix("d");
		}
		
		/*
		 * General quadratic programming:
		 *
		 *      min 2 \ x' * Q * x + c' * x
		 * s.t. A * x = b
		 *      B * x <= d
		 */
		GeneralQP.solve(Q, c, A, b, B, d);

	}
	
	/**
	 * Solve a general quadratic programming problem formulated as
	 * <p>
	 *      min 2 \ x' * Q * x + c' * x </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * 
	 * @param Q an n x n positive definite or semi-definite matrix
	 * 
	 * @param c an n x 1 real matrix
	 * 
	 * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     * 
	 * @return a {@code QPSolution} instance if the general QP problems
	 *         is feasible or null otherwise
	 *         
	 */
	public static QPSolution solve(Matrix Q, Matrix c, Matrix A, Matrix b, Matrix B, Matrix d) {

		fprintf("Phase I:\n\n");
		PhaseIResult phaseIResult = phaseI(A, b, B, d);
		if (phaseIResult.feasible) {
			fprintf("Phase II:\n\n");
			Matrix x0 = phaseIResult.optimizer;
			return phaseII(Q, c, A, b, B, d, x0);
		} else {
			System.err.println("The QP problem is infeasible!\n");
			return null;
		}
		
	}
	
	/**
	 * We demonstrate the implementation of phase I via primal-dual interior
	 * point method to test whether the following problem is feasible:
	 * </p>
	 *      min f(x) </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * We seek the optimizer for the following phase I problem:
	 * </p>
     *      min 1's </br>
     * s.t. A * x = b </br>
     *      B * x - d <= s </br>
     *      s >= 0 </br>
     * </p>     
     * <=> </br>
     *      min cI'y </br>
     * s.t. AI * y = b </br>
     *      BI * y <= dI </br>
     * </p>
     * cI = [zeros(n, 1); ones(m, 1)] </br>
     * AI = [A zeros(p, m)] </br>
     * BI = [B -eye(m); zeros(m, n) -eye(m)] </br>
     * dI = [d; zeros(m, 1)] </br>
     * y = [x; s] </br>
     * 
     * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     *      
	 * @return a {@code PhaseIResult} instance if feasible or null if infeasible
	 * 
	 */
	public static PhaseIResult phaseI(Matrix A, Matrix b, Matrix B, Matrix d) {
		
		/*
		 * Number of unknown variables
		 */
		int n = A.getColumnDimension();
		
		/*
		 * Number of equality constraints
		 */
		int p = A.getRowDimension();
		
		/*
		 * Number of inequality constraints
		 */
		int m = B.getRowDimension();
		
		Matrix A_ori = A;
		Matrix B_ori = B;
		Matrix d_ori = d;
		
		Matrix c = vertcat(zeros(n, 1), ones(m, 1));
		A = horzcat(A, zeros(p, m));
		B = vertcat(horzcat(B, uminus(eye(m))), horzcat(zeros(m, n), uminus(eye(m))));
		d = vertcat(d, zeros(m, 1));
		
		int n_ori = n;
		int m_ori = m;
		
		n = n + m;
		m = 2 * m;
		
		// Matrix x0 = rand(n_ori, 1);
		Matrix x0 = ones(n_ori, 1);
		Matrix s0 = B_ori.mtimes(x0).minus(d_ori).plus(ones(m_ori, 1));
		x0 = vertcat(x0, s0);
		Matrix v0 = zeros(p, 1);
		
		// Parameter setting
		
		double mu = 1.8;
		double epsilon = 1e-10;
		double epsilon_feas = 1e-10;
		double alpha = 0.1;
		double beta = 0.98;
		
		tic();
		
		Matrix l0 = rdivide(ones(m, 1), m);

		Matrix x = x0;
		Matrix l = l0;
		Matrix v = v0;
		
		Matrix F_x_0 = B.mtimes(x).minus(d);
		
		double eta_t = -innerProduct(F_x_0, l0);
		double t = 1;
		double f_x = 0;
		Matrix G_f_x = null;
		Matrix F_x = null;
		Matrix DF_x = null;
		Matrix H_x = times(1e-10, eye(n));
		Matrix r_prim = null;
		Matrix r_dual = null;
		Matrix r_cent = null;
		Matrix Matrix = null;
		Matrix Vector = null;
		
		double residual = 0;
		double residual_prim = 0;
		double residual_dual = 0;
		
		Matrix z_pd = null;
		Matrix x_nt = null;
		Matrix l_nt = null;
		Matrix v_nt = null;
		
		Matrix x_s = zeros(size(x0));
		Matrix l_s = zeros(size(l0));
		Matrix v_s = zeros(size(v0));
		
		double s = 0;
		Matrix G_f_x_s = null;
        Matrix F_x_s = null;
        Matrix DF_x_s = null;
        
        Matrix r_prim_s = null;
        Matrix r_dual_s = null;
		Matrix r_cent_s = null;
        double residual_s = 0;
        // int k = 0;
		while (true) {

			t = mu * m / eta_t;
	    	f_x = innerProduct(c, x);
	    	
	    	// Calculate the gradient of f(x)
	    	G_f_x = c;
	    
	    	// Calculate F(x) and DF(x)
	    	F_x = B.mtimes(x).minus(d);
	    	DF_x = B;
	    	
	    	// Calculate the Hessian matrix of f(x) and fi(x)
	    	// H_x = times(1e-10, eye(n));
	    	
	    	r_prim = A.mtimes(x).minus(b);
	        r_dual = G_f_x.plus(DF_x.transpose().mtimes(l)).plus(A.transpose().mtimes(v));
	        r_cent = uminus(times(l, F_x)).minus(rdivide(ones(m, 1), t));
	        
	        Matrix = vertcat(
	        			horzcat(H_x, DF_x.transpose(), A.transpose()),
	        			horzcat(uminus(mtimes(diag(l), DF_x)), uminus(diag(F_x)), zeros(m, p)),
	        			horzcat(A, zeros(p, m), zeros(p, p))
	        		);    	
	        Vector = uminus(vertcat(r_dual, r_cent, r_prim));
	    
	        residual = norm(Vector);
	        residual_prim = norm(r_prim);
	        residual_dual = norm(r_dual);
	        eta_t = - innerProduct(F_x, l);
	        
	        // fprintf("f_x: %g, residual: %g\n", f_x, residual);
	        if (residual_prim <= epsilon_feas &&
	        	residual_dual <= epsilon_feas &&
	        	eta_t <= epsilon) {
	        	fprintf("Terminate successfully.\n\n");
	        	break;
	        }
	    	
	        z_pd = mldivide(Matrix, Vector);
	        /*fprintf("k = %d%n", k++);
	        disp(z_pd.transpose());*/
	        /*x_nt = z_pd.getSubMatrix(0, n - 1, 0, 0);
	        l_nt = z_pd.getSubMatrix(n, n + m - 1, 0, 0);
	        v_nt = z_pd.getSubMatrix(n + m, n + m + p - 1, 0, 0);*/
	        x_nt = getRows(z_pd, 0, n - 1);
	        l_nt = getRows(z_pd, n, n + m - 1);
	        v_nt = getRows(z_pd, n + m, n + m + p - 1);
	        
	        // Backtracking line search
	        
	        s = 1;
	        // Ensure lambda to be nonnegative
	        while (true) {
	            // l_s = plus(l, times(s, l_nt));
	        	affine(l_s, s, l_nt, '+', l);
	            if (sumAll(lt(l_s, 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        // Ensure f_i(x) <= 0, i = 1, 2, ..., m
	        while (true) {
	            // x_s = plus(x, times(s, x_nt));
	        	affine(x_s, s, x_nt, '+', x);
	            if (sumAll(lt(d.minus(B.mtimes(x_s)), 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        while (true) {
	        	
	        	/*x_s = plus(x, times(s, x_nt));
	        	l_s = plus(l, times(s, l_nt));
	        	v_s = plus(v, times(s, v_nt));*/
	        	affine(x_s, s, x_nt, '+', x);
	        	affine(l_s, s, l_nt, '+', l);
	        	affine(v_s, s, v_nt, '+', v);
		        
		        // Template {
		        
		        // Calculate the gradient of f(x_s)
		        G_f_x_s = c;
		        
		        // Calculate F(x_s) and DF(x_s)
		        F_x_s = B.mtimes(x_s).minus(d);
		        DF_x_s = B;
		        
		        // }
		        
		        r_prim_s = A.mtimes(x_s).minus(b);
		        r_dual_s = G_f_x_s.plus(DF_x_s.transpose().mtimes(l_s)).plus(A.transpose().mtimes(v_s));
		        r_cent_s = uminus(times(l_s, F_x_s)).minus(rdivide(ones(m, 1), t));
		         
		        residual_s = norm(vertcat(r_dual_s, r_cent_s, r_prim_s));
		        if (residual_s <= (1 - alpha * s) * residual)
		            break;
		        else
		            s = beta * s;
		        
		    }
	        
	        /*x = x_s;
	        l = l_s;
	        v = v_s;*/
	        assign(x, x_s);
	        assign(l, l_s);
	        assign(v, v_s);
	    	
		}
		
		double t_sum_of_inequalities = toc();

		Matrix x_opt = getRows(x, 0, n_ori - 1);
		fprintf("x_opt:\n");
		disp(x_opt.transpose());

		Matrix s_opt = getRows(x, n_ori, n - 1);
		fprintf("s_opt:\n");
		disp(s_opt.transpose());

		Matrix lambda_s = getRows(l, m_ori, m - 1);
		fprintf("lambda for the inequalities s_i >= 0:\n");
		disp(lambda_s.transpose());

		Matrix e = B_ori.mtimes(x_opt).minus(d_ori);
		fprintf("B * x - d:\n");
		disp(e.transpose());

		Matrix lambda_ineq = getRows(l, 0, m_ori - 1);
		fprintf("lambda for the inequalities fi(x) <= s_i:\n");
		disp(lambda_ineq.transpose());

		Matrix v_opt = v;
		fprintf("nu for the equalities A * x = b:\n");
		disp(v_opt.transpose());

		fprintf("residual: %g\n\n", residual);
		fprintf("A * x - b:\n");
		disp(A_ori.mtimes(x_opt).minus(b).transpose());
		fprintf("norm(A * x - b, \"fro\"): %f\n\n", norm(A_ori.mtimes(x_opt).minus(b), "fro"));

		double fval_opt = f_x;
		fprintf("fval_opt: %g\n\n", fval_opt);
		boolean feasible = false;
		if (fval_opt <= epsilon) {
		    feasible = true;
		    fprintf("The problem is feasible.\n\n");
		} else {
		    feasible = false;
		    fprintf("The problem is infeasible.\n\n");
		}
		fprintf("Computation time: %f seconds\n\n", t_sum_of_inequalities);

		/*if (!feasible)
		    return null;*/

		x0 = x_opt;

		int pause_time = 1;
		fprintf("halt execution temporarily in %d seconds...\n\n", pause_time);
		pause(pause_time);
		
		return new PhaseIResult(feasible, x_opt, fval_opt);
		
	}
	
	/**
	 * Phase II for solving a general quadratic programming problem formulated as
	 * <p>
	 *      min 2 \ x' * Q * x + c' * x </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * 
	 * @param Q an n x n positive definite or semi-definite matrix
	 * 
	 * @param c an n x 1 real matrix
	 * 
	 * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     * 
     * @param x0 starting point
     * 
	 * @return a {@code QPSolution} instance
	 *         
	 */
	public static QPSolution phaseII(Matrix Q, Matrix c, Matrix A, Matrix b, Matrix B, Matrix d, Matrix x0) {
		
		/*
		 * Number of unknown variables
		 */
		int n = A.getColumnDimension();
		
		/*
		 * Number of equality constraints
		 */
		int p = A.getRowDimension();
		
		/*
		 * Number of inequality constraints
		 */
		int m = B.getRowDimension();
		
		
		Matrix v0 = zeros(p, 1);
		
		// Parameter setting
		
		double mu = 1.8;
		double epsilon = 1e-10;
		double epsilon_feas = 1e-10;
		double alpha = 0.1;
		double beta = 0.98;
		
		tic();
		
		Matrix l0 = rdivide(ones(m, 1), m);

		Matrix x = x0;
		Matrix l = l0;
		Matrix v = v0;
		
		Matrix F_x_0 = B.mtimes(x).minus(d);
		
		double eta_t = - innerProduct(F_x_0, l0);
		double t = 1;
		double f_x = 0;
		Matrix G_f_x = null;
		Matrix F_x = null;
		Matrix DF_x = null;
		Matrix H_x = Q;
		Matrix r_prim = null;
		Matrix r_dual = null;
		Matrix r_cent = null;
		Matrix Matrix = null;
		Matrix Vector = null;
		
		double residual = 0;
		double residual_prim = 0;
		double residual_dual = 0;
		
		Matrix z_pd = null;
		Matrix x_nt = null;
		Matrix l_nt = null;
		Matrix v_nt = null;
		
		/*Matrix x_s = null;
		Matrix l_s = null;
		Matrix v_s = null;*/
		Matrix x_s = zeros(size(x0));
		Matrix l_s = zeros(size(l0));
		Matrix v_s = zeros(size(v0));
		
		double s = 0;
		Matrix G_f_x_s = null;
        Matrix F_x_s = null;
        Matrix DF_x_s = null;
        
        Matrix r_prim_s = null;
        Matrix r_dual_s = null;
		Matrix r_cent_s = null;
        double residual_s = 0;
        
		while (true) {

			t = mu * m / eta_t;
	    	f_x = innerProduct(x, Q.mtimes(x)) / 2 + innerProduct(c, x);
	    	
	    	// Calculate the gradient of f(x)
	    	G_f_x = Q.mtimes(x).plus(c);
	    
	    	// Calculate F(x) and DF(x)
	    	F_x = B.mtimes(x).minus(d);
	    	DF_x = B;
	    	
	    	// Calculate the Hessian matrix of f(x) and fi(x)
	    	// H_x = times(1e-10, eye(n));
	    	
	    	r_prim = A.mtimes(x).minus(b);
	        r_dual = G_f_x.plus(DF_x.transpose().mtimes(l)).plus(A.transpose().mtimes(v));
	        r_cent = uminus(times(l, F_x)).minus(rdivide(ones(m, 1), t));
	        
	        Matrix = vertcat(
	        			horzcat(H_x, DF_x.transpose(), A.transpose()),
	        			horzcat(uminus(mtimes(diag(l),DF_x)), uminus(diag(F_x)), zeros(m, p)),
	        			horzcat(A, zeros(p, m), zeros(p, p))
	        		);    	
	        Vector = uminus(vertcat(r_dual, r_cent, r_prim));
	    
	        residual = norm(Vector);
	        residual_prim = norm(r_prim);
	        residual_dual = norm(r_dual);
	        eta_t = -innerProduct(F_x, l);
	        
	        // fprintf("f_x: %g, residual: %g\n", f_x, residual);
	        if (residual_prim <= epsilon_feas &&
	        	residual_dual <= epsilon_feas &&
	        	eta_t <= epsilon) {
	        	fprintf("Terminate successfully.\n\n");
	        	break;
	        }
	    	
	        z_pd = mldivide(Matrix, Vector);
	        /*x_nt = z_pd.getSubMatrix(0, n - 1, 0, 0);
	        l_nt = z_pd.getSubMatrix(n, n + m - 1, 0, 0);
	        v_nt = z_pd.getSubMatrix(n + m, n + m + p - 1, 0, 0);*/
	        x_nt = getRows(z_pd, 0, n - 1);
	        l_nt = getRows(z_pd, n, n + m - 1);
	        v_nt = getRows(z_pd, n + m, n + m + p - 1);
	        
	        // Backtracking line search
	        
	        s = 1;
	        // Ensure lambda to be nonnegative
	        while (true) {
	            // l_s = plus(l, times(s, l_nt));
	        	affine(l_s, s, l_nt, '+', l);
	            if (sumAll(lt(l_s, 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        // Ensure f_i(x) <= 0, i = 1, 2, ..., m
	        while (true) {
	            // x_s = plus(x, times(s, x_nt));
	        	affine(x_s, s, x_nt, '+', x);
	            if (sumAll(lt(d.minus(B.mtimes(x_s)), 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        while (true) {
	        	
	        	/*x_s = plus(x, times(s, x_nt));
	        	l_s = plus(l, times(s, l_nt));
	        	v_s = plus(v, times(s, v_nt));*/
	        	affine(x_s, s, x_nt, '+', x);
	        	affine(l_s, s, l_nt, '+', l);
	        	affine(v_s, s, v_nt, '+', v);
		        
		        // Template {
		        
		        // Calculate the gradient of f(x_s)
		        G_f_x_s = Q.mtimes(x_s).plus(c);
		        
		        // Calculate F(x_s) and DF(x_s)
		        F_x_s = B.mtimes(x_s).minus(d);
		        DF_x_s = B;
		        
		        // }
		        
		        r_prim_s = A.mtimes(x_s).minus(b);
		        r_dual_s = G_f_x_s.plus(DF_x_s.transpose().mtimes(l_s)).plus(A.transpose().mtimes(v_s));
		        r_cent_s = uminus(times(l_s, F_x_s)).minus(rdivide(ones(m, 1), t));
		         
		        residual_s = norm(vertcat(r_dual_s, r_cent_s, r_prim_s));
		        if (residual_s <= (1 - alpha * s) * residual)
		            break;
		        else
		            s = beta * s;
		        
		    }
	        
	        /*x = x_s;
	        l = l_s;
	        v = v_s;*/
	        assign(x, x_s);
	        assign(l, l_s);
	        assign(v, v_s);
	    	
		}
		
		double t_primal_dual_interior_point = toc();
		
		double fval_primal_dual_interior_point = f_x;
		Matrix x_primal_dual_interior_point = x;
		Matrix lambda_primal_dual_interior_point = l;
		Matrix v_primal_dual_interior_point = v;

		fprintf("residual: %g\n\n", residual);
		fprintf("Optimal objective function value: %g\n\n", fval_primal_dual_interior_point);
		fprintf("Optimizer:\n");
		disp(x_primal_dual_interior_point.transpose());

		Matrix e = B.mtimes(x).minus(d);
		fprintf("B * x - d:\n");
		disp(e.transpose());

		fprintf("lambda:\n");
		disp(lambda_primal_dual_interior_point.transpose());

		fprintf("nu:\n");
		disp(v_primal_dual_interior_point.transpose());

		fprintf("norm(A * x - b, \"fro\"): %f\n\n", norm(A.mtimes(x_primal_dual_interior_point).minus(b), "fro"));
		fprintf("Computation time: %f seconds\n\n", t_primal_dual_interior_point);
		
		return new QPSolution(x, l, v, f_x);
		
	}

}
