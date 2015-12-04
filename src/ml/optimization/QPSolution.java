package ml.optimization;

import la.matrix.Matrix;

public class QPSolution {

	public Matrix optimizer;
	
	public Matrix lambda_opt;
	
	public Matrix nu_opt;
	
	public double optimum;
	
	public QPSolution(Matrix optimizer, Matrix lambda_opt, Matrix nu_opt, double optimum) {
		this.optimizer = optimizer;
		this.lambda_opt = lambda_opt;
		this.nu_opt = nu_opt;
		this.optimum = optimum;
	}
	
}
