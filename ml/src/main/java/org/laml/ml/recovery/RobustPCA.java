package org.laml.ml.recovery;

import static org.laml.ml.optimization.ShrinkageOperator.shrinkage;
import static org.laml.la.utils.InPlaceOperator.plusAssign;
import static org.laml.la.utils.Matlab.abs;
import static org.laml.la.utils.Matlab.max;
import static org.laml.la.utils.Matlab.minus;
import static org.laml.la.utils.Matlab.mtimes;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rand;
import static org.laml.la.utils.Matlab.randn;
import static org.laml.la.utils.Matlab.randperm;
import static org.laml.la.utils.Matlab.rank;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.reshape;
import static org.laml.la.utils.Matlab.setSubMatrix;
import static org.laml.la.utils.Matlab.size;
import static org.laml.la.utils.Matlab.svd;
import static org.laml.la.utils.Matlab.vec;
import static org.laml.la.utils.Matlab.zeros;
import static org.laml.la.utils.Printer.disp;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Time.tic;
import static org.laml.la.utils.Time.toc;
import org.laml.la.matrix.Matrix;

/***
 * A Java implementation of robust PCA which solves the 
 * following convex optimization problem:
 * </br>
 * min ||A||_* + lambda ||E||_1</br>
 * s.t. D = A + E</br>
 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
 * the sum of its singular values), and ||.||_1 denotes the 
 * sum of the absolute values of matrix entries.</br>
 * </br>
 * Inexact augmented Lagrange multipliers is used to solve the optimization
 * problem due to its empirically fast convergence speed and proved convergence 
 * to the true optimal solution.
 * </br>
 * <b>Input:</b></br>
 *    D: an observation matrix</br>
 *    lambda: a positive weighting parameter</br>
 *    
 * <b>Output:</b></br>
 *    A: a low-rank matrix recovered from the corrupted data matrix D</br>
 *    E: error matrix between D and A</br>
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 2nd, 2014
 */
public class RobustPCA {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 8;
		int r = m / 4;
		
		Matrix L = randn(m, r);
		Matrix R = randn(m, r);
		
		Matrix A_star = mtimes(L, R.transpose());
		Matrix E_star = zeros(size(A_star));
		int[] indices = randperm(m * m);
		int nz = m * m / 20;
		int[] nz_indices = new int[nz];
		for (int i = 0; i < nz; i++) {
			nz_indices[i] = indices[i] - 1;
		}
		Matrix E_vec = vec(E_star);
		setSubMatrix(E_vec, nz_indices, new int[] {0}, (minus(rand(nz, 1), 0.5).times(100)));
		E_star = reshape(E_vec, size(E_star));
		// disp(E_star);
		// Input
		Matrix D = A_star.plus(E_star);
		double lambda = 1 * Math.pow(m, -0.5);
		
		/*// Test
		double[][] data = new double[][]{
                {10.1,19.6,28.3,9.7},
                {9.5,20.5,28.9,10.1},
                {10.7,20.2,31,10.3},
                {9.9,21.5,31.7,9.5},
                {10.3,21.1,31.1,10},
                {10.8,20.4,29.2,10},
                {10.5,20.9,29.1,10.8},
                {9.9,19.6,28.8,10.3},
                {9.7,20.7,31,9.6},
                {9.3,19.7,30.3,9.9},
                {11,24,35,-0.2},
                {12,23,37,-0.4},
                {12,26,34,0.7},
                {11,34,34,0.1},
                {3.4,2.9,2.1,-0.4},
                {3.1,2.2,0.3,0.6},
                {0,1.6,0.2,-0.2},
                {2.3,1.6,2,0},
                {0.8,2.9,1.6,0.1},
                {3.1,3.4,2.2,0.4},
                {2.6,2.2,1.9,0.9},
                {0.4,3.2,1.9,0.3},
                {2,2.3,0.8,-0.8},
                {1.3,2.3,0.5,0.7},
                {1,0,0.4,-0.3},
                {0.9,3.3,2.5,-0.8},
                {3.3,2.5,2.9,-0.7},
                {1.8,0.8,2,0.3},
                {1.2,0.9,0.8,0.3},
                {1.2,0.7,3.4,-0.3},
                {3.1,1.4,1,0},
                {0.5,2.4,0.3,-0.4},
                {1.5,3.1,1.5,-0.6},
                {0.4,0,0.7,-0.7},
                {3.1,2.4,3,0.3},
                {1.1,2.2,2.7,-1},
                {0.1,3,2.6,-0.6},
                {1.5,1.2,0.2,0.9},
                {2.1,0,1.2,-0.7},
                {0.5,2,1.2,-0.5},
                {3.4,1.6,2.9,-0.1},
                {0.3,1,2.7,-0.7},
                {0.1,3.3,0.9,0.6},
                {1.8,0.5,3.2,-0.7},
                {1.9,0.1,0.6,-0.5},
                {1.8,0.5,3,-0.4},
                {3,0.1,0.8,-0.9},
                {3.1,1.6,3,0.1},
                {3.1,2.5,1.9,0.9},
                {2.1,2.8,2.9,-0.4},
                {2.3,1.5,0.4,0.7},
                {3.3,0.6,1.2,-0.5},
                {0.3,0.4,3.3,0.7},
                {1.1,3,0.3,0.7},
                {0.5,2.4,0.9,0},
                {1.8,3.2,0.9,0.1},
                {1.8,0.7,0.7,0.7},
                {2.4,3.4,1.5,-0.1},
                {1.6,2.1,3,-0.3},
                {0.3,1.5,3.3,-0.9},
                {0.4,3.4,3,-0.3},
                {0.9,0.1,0.3,0.6},
                {1.1,2.7,0.2,-0.3},
                {2.8,3,2.9,-0.5},
                {2,0.7,2.7,0.6},
                {0.2,1.8,0.8,-0.9},
                {1.6,2,1.2,-0.7},
                {0.1,0,1.1,0.6},
                {2,0.6,0.3,0.2},
                {1,2.2,2.9,0.7},
                {2.2,2.5,2.3,0.2},
                {0.6,2,1.5,-0.2},
                {0.3,1.7,2.2,0.4},
                {0,2.2,1.6,-0.9},
                {0.3,0.4,2.6,0.2}
                };
		D = new DenseMatrix(data);
		lambda = 1;*/
		
		// Run Robust PCA
		RobustPCA robustPCA = new RobustPCA(lambda);
		robustPCA.feedData(D);
		tic();
		robustPCA.run();
		fprintf("Elapsed time: %.2f seconds.%n", toc());
		
		// Output
		Matrix A_hat = robustPCA.GetLowRankEstimation();
		Matrix E_hat = robustPCA.GetErrorMatrix();
		
		fprintf("A*:\n");
		disp(A_star, 4);
		fprintf("A^:\n");
		disp(A_hat, 4);
		fprintf("E*:\n");
		disp(E_star, 4);
		fprintf("E^:\n");
		disp(E_hat, 4);
		fprintf("rank(A*): %d\n", rank(A_star));
		fprintf("rank(A^): %d\n", rank(A_hat));
		fprintf("||A* - A^||_F: %.4f\n", norm(A_star.minus(A_hat), "fro"));
		fprintf("||E* - E^||_F: %.4f\n", norm(E_star.minus(E_hat), "fro"));
		
	}
	
	/**
	 * A positive weighting parameter.
	 */
	double lambda;

	/**
	 * Observation real matrix.
	 */
	Matrix D;
	
	/**
	 * A low-rank matrix recovered from the corrupted data observation matrix D.
	 */
	Matrix A;
	
	/**
	 * Error matrix between the original observation matrix D and the low-rank
	 * recovered matrix A.
	 */
	Matrix E;
	
	/**
	 * Constructor for Robust PCA.
	 * 
	 * @param lambda a positive weighting parameter, larger value leads to sparser
	 * 				 error matrix
	 */
	public RobustPCA(double lambda) {
		this.lambda = lambda;
	}
	
	/**
	 * Feed an observation matrix.
	 * 
	 * @param D a real matrix
	 */
	public void feedData(Matrix D) {
		this.D = D;
	}
	
	/**
	 * Run robust PCA.
	 */
	public void run() {
		Matrix[] res = run(D, lambda);
		A = res[0];
		E = res[1];
	}
	
	/**
	 * Get the low-rank matrix recovered from the corrupted data 
	 * observation matrix.
	 * 
	 * @return low-rank approximation
	 */
	public Matrix GetLowRankEstimation() {
		return A;
	}
	
	/**
	 * Get the error matrix between the original observation matrix 
	 * and its low-rank recovered matrix.
	 * 
	 * @return error matrix
	 */
	public Matrix GetErrorMatrix() {
		return E;
	}
	
	/**
	 * Compute robust PCA for an observation matrix which solves the 
	 * following convex optimization problem:
	 * </br>
	 * min ||A||_* + lambda ||E||_1</br>
	 * s.t. D = A + E</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values), and ||.||_1 denotes the 
	 * sum of the absolute values of matrix entries.</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param lambda a positive weighting parameter, larger value leads to sparser
	 * 				 error matrix
	 * @return a {@code Matrix} array [A, E] where A is the recovered low-rank
	 * 		   approximation of D, and E is the error matrix between A and D
	 * 
	 */
	static public Matrix[] run(Matrix D, double lambda) {
		
		Matrix Y = rdivide(D, J(D, lambda));
		Matrix E = zeros(size(D));
		Matrix A = minus(D, E);
		/*fprintf("norm(D, inf) = %f\n", norm(D, Matlab.inf));
		fprintf("norm(D, 2) = %f\n", norm(D, 2));*/
		double mu = 1.25 / norm(D, 2);
		double rou = 1.6;
		int k = 0;
		double norm_D = norm(D, "fro");
		double e1 = 1e-7;
		double e2 = 1e-6;
		double c1 = 0;
		double c2 = 0;
		double mu_old = 0;
		Matrix E_old = null;
		Matrix[] SVD = null;
		
		while (true) {
			
			// Stopping criteria
			if (k > 0) {
				c1 = norm(D.minus(A).minus(E), "fro") / norm_D;
				c2 = mu_old * norm(E.minus(E_old), "fro") / norm_D;
				// fprintf("k = %d, c2: %.4f%n", k, c2);
				if (c1 <= e1 && c2 <= e2)
					break;
			}
			
			E_old = E;
			mu_old = mu;
			
			// E_{k+1} = argmin_E L(A_k, E, Y_k, mu_k)
		    shrinkage(E, plus(minus(D, A), rdivide(Y, mu)), lambda / mu);
		    
		    // A_{k+1} = argmin_A L(A, E_{k+1}, Y_k, mu_k)
		    SVD = svd(plus(minus(D, E), rdivide(Y, mu)));
		    /*disp(SVD[1]);
		    disp(shrinkage(SVD[1], 1 / mu));*/
		    A = SVD[0].mtimes(shrinkage(SVD[1], 1 / mu)).mtimes(SVD[2].transpose());

		    // Y = Y.plus(times(mu, D.minus(A).minus(E)));
		    plusAssign(Y, mu, D.minus(A).minus(E));
		    
		    // Update mu_k to mu_{k+1}

		    if (norm(E.minus(E_old), "fro") * mu / norm_D < e2)
		    	mu = rou * mu;

		    k = k + 1;

		}

		Matrix[] res = new Matrix[2];
		res[0] = A;
		res[1] = E;
		return res;
		
	}
	
	private static double J(Matrix Y, double lambda) {
		return Math.max(norm(Y, 2), max(max(abs(Y))[0])[0] / lambda);
	}

}
