package ml.optimization;

import static la.io.IO.loadMatrix;
import static la.io.IO.saveMatrix;
import static la.utils.Matlab.eye;
import static la.utils.Matlab.innerProduct;
import static la.utils.Matlab.norm;
import static la.utils.Matlab.ones;
import static la.utils.Matlab.plus;
import static la.utils.Matlab.rand;
import static la.utils.Matlab.times;
import static la.utils.Printer.disp;
import static la.utils.Printer.fprintf;
import static la.utils.Time.tic;
import static la.utils.Time.toc;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;

/**
 * General quadratic programming:
 * <p>
 *      min 2 \ x' * Q * x + c' * x </br>
 * s.t. A * x = b </br>
 *      B * x <= d </br>
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 27th, 2014
 */
public class GeneralQPViaPrimalDualInteriorPoint {

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
		GeneralQPViaPrimalDualInteriorPoint.solve(Q, c, A, b, B, d);
		
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
		PhaseIResult phaseIResult = GeneralQP.phaseI(A, b, B, d);
		if (phaseIResult.feasible) {
			fprintf("Phase II:\n\n");
			Matrix x0 = phaseIResult.optimizer;
			// GeneralQP.phaseII(Q, c, A, b, B, d, x0);
			return phaseII(Q, c, A, b, B, d, x0);
		} else {
			System.err.println("The QP problem is infeasible!\n");
			return null;
		}
		
	}

	private static QPSolution phaseII(Matrix Q, Matrix c, Matrix A,
			Matrix b, Matrix B, Matrix d, Matrix x0) {
		
		Matrix x = x0.copy();
		Matrix l = new DenseMatrix(B.getRowDimension(), 1);
		Matrix v = new DenseMatrix(A.getRowDimension(), 1);
		Matrix H_x = null;
		Matrix F_x = null;
		Matrix DF_x = null;
		Matrix G_f_x = null;
		double fval = 0;
		
		fval = innerProduct(x, Q.mtimes(x)) / 2 + innerProduct(c, x);
		H_x = Q;
		DF_x = B;
		F_x = B.mtimes(x).minus(d);
		G_f_x = Q.mtimes(x).plus(c);
		
		boolean flags[] = null;
		int k = 0;
		// int maxIter = 1000;
		tic();
		while (true) {
			flags = PrimalDualInteriorPoint.run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, l, v);
			/*fprintf("F_x");
			display(F_x.transpose());
			fprintf("DF_x");
			display(DF_x);
			fprintf("G_f_x");
			display(G_f_x.transpose());
			fprintf("x");
			display(x.transpose());*/

			/*if (toc() > 3) {
				int a = 1;
				a = a + 1;
			}*/
			
			if (flags[0])
				break;
			
			/*
			 *  Compute the objective function value, if flags[1] is true
			 *  gradient will also be computed.
			 */
			fval = innerProduct(x, Q.mtimes(x)) / 2 + innerProduct(c, x);
			F_x = B.mtimes(x).minus(d);
			// disp(F_x.transpose());
			// G_f_x = Q.mtimes(x).plus(c);
			if (flags[1]) {
				k = k + 1;
				// Compute the gradient
				/*if (k > maxIter)
					break;*/
				
				G_f_x = Q.mtimes(x).plus(c);
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
			}
			
		}
		
		double t_primal_dual_interior_point = toc();
		
		double fval_primal_dual_interior_point = fval;
		Matrix x_primal_dual_interior_point = x;
		Matrix lambda_primal_dual_interior_point = l;
		Matrix v_primal_dual_interior_point = v;

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
		
		return new QPSolution(x, l, v, fval);
		
	}

}
