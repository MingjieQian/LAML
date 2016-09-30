package ml.optimization;

import static la.utils.Matlab.eye;
import static la.utils.Matlab.innerProduct;
import static la.utils.Matlab.minus;
import static la.utils.Matlab.plus;
import static la.utils.Matlab.rand;
import static la.utils.Matlab.times;
import static la.utils.Matlab.zeros;
import static la.utils.Printer.disp;
import static la.utils.Printer.display;
import static la.utils.Printer.fprintf;
import la.matrix.Matrix;

/**
 * Quadratic programming with bound constraints:
 * <p>
 *      min 2 \ x' * Q * x + c' * x
 *      s.t. l <= x <= u
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 25th, 2014
 */
public class QPWithBoundConstraints {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int n = 5;
		Matrix x = rand(n);
		Matrix Q = minus(x.mtimes(x.transpose()), times(rand(1).getEntry(0, 0), eye(n)));
		Matrix c = plus(-2, times(2, rand(n, 1)));
		double l = 0;
		double u = 1;
		double epsilon = 1e-6;
		
		QPSolution S = QPWithBoundConstraints.solve(Q, c, l, u, epsilon);
		
		disp("Q:");
		disp(Q);
		disp("c:");
		disp(c);
		fprintf("Optimum: %g\n", S.optimum);
		fprintf("Optimizer:\n");
		display(S.optimizer.transpose());
		
	}
	
	/**
	 * Solve this bound constrained QP problem.
	 * 
	 * @param Q the positive semi-definite matrix
	 * 
	 * @param c the linear coefficient vector
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @return a {@code QPSolution} instance containing the optimizer
	 *           and the optimum
	 * 
	 */
	public static QPSolution solve(Matrix Q, Matrix c, double l, double u, double epsilon) {
		return solve(Q, c, l, u, epsilon, null);
	}
	
	/**
	 * Solve this bound constrained QP problem.
	 * 
	 * @param Q the positive semi-definite matrix
	 * 
	 * @param c the linear coefficient vector
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param x0 staring point if not null
	 * 
	 * @return a {@code QPSolution} instance containing the optimizer
	 *           and the optimum
	 * 
	 */
	public static QPSolution solve(Matrix Q, Matrix c, double l, double u, double epsilon, Matrix x0) {
		
		int d = Q.getColumnDimension();
		double fval = 0;
		Matrix x = null;
		if (x0 != null) {
			x = x0;
		} else {
			x = plus((l + u) / 2, zeros(d, 1));
		}
		
		/* 
		 * Grad = Q * x + c
		 */
		Matrix Grad = Q.mtimes(x).plus(c);
		fval = innerProduct(x, Q.mtimes(x)) / 2 + innerProduct(c, x);
		
		boolean flags[] = null;
		while (true) {
			flags = BoundConstrainedPLBFGS.run(Grad, fval, l, u, epsilon, x);
			if (flags[0])
				break;
			fval = innerProduct(x, Q.mtimes(x)) / 2 + innerProduct(c, x);
			if (flags[1])
				Grad = Q.mtimes(x).plus(c);
		}
		return new QPSolution(x, null, null, fval);
		
	}
	
}
