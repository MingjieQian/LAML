package org.laml.ml.recovery;

import static org.laml.ml.optimization.ShrinkageOperator.shrinkage;
import static org.laml.la.utils.ArrayOperator.colon;
import static org.laml.la.utils.ArrayOperator.minusAssign;
import static org.laml.la.utils.InPlaceOperator.assign;
import static org.laml.la.utils.InPlaceOperator.minus;
import static org.laml.la.utils.InPlaceOperator.plusAssign;
import static org.laml.la.utils.Matlab.gt;
import static org.laml.la.utils.Matlab.linearIndexing;
import static org.laml.la.utils.Matlab.linearIndexingAssignment;
import static org.laml.la.utils.Matlab.logicalIndexingAssignment;
import static org.laml.la.utils.Matlab.minus;
import static org.laml.la.utils.Matlab.mtimes;
import static org.laml.la.utils.Matlab.norm;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.randn;
import static org.laml.la.utils.Matlab.randperm;
import static org.laml.la.utils.Matlab.rank;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.size;
import static org.laml.la.utils.Matlab.sumAll;
import static org.laml.la.utils.Matlab.svd;
import static org.laml.la.utils.Matlab.zeros;
import static org.laml.la.utils.Printer.disp;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Time.tic;
import static org.laml.la.utils.Time.toc;
import org.laml.la.io.IO;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;

/***
 * A Java implementation of matrix completion which solves the 
 * following convex optimization problem:
 * </br>
 * min ||A||_*</br>
 * s.t. D = A + E</br>
 *      E(Omega) = 0</br>
 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
 * the sum of its singular values).</br>
 * </br>
 * Inexact augmented Lagrange multiplers is used to solve the optimization
 * problem due to its empirically fast convergence speed and proved convergence 
 * to the true optimal solution.
 * 
 * <b>Input:</b></br>
 *    D: an observation matrix with columns as data vectors</br>
 *    Omega: a sparse or dense logical matrix indicating the indices of samples</br>
 *    
 * <b>Output:</b></br>
 *    A: a low-rank matrix completed from the data matrix D</br>
 *    E: error matrix between D and A</br>
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 2nd, 2014
 */
public class MatrixCompletion {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 6;
		int r = 1;
		int p = (int) Math.round(m * m * 0.3);
		
		Matrix L = randn(m, r);
		Matrix R = randn(m, r);
		Matrix A_star = mtimes(L, R.transpose());
		
		int[] indices = randperm(m * m);
		minusAssign(indices, 1);
		indices = linearIndexing(indices, colon(0, p - 1));
		
		Matrix Omega = zeros(size(A_star));
		linearIndexingAssignment(Omega, indices, 1);
		
		Matrix D = zeros(size(A_star));
		linearIndexingAssignment(D, indices, linearIndexing(A_star, indices));
				
		Matrix E_star = D.minus(A_star);
		logicalIndexingAssignment(E_star, Omega, 0);
		
		D = IO.loadMatrix("D.txt");
		Omega = IO.loadMatrix("Omega.txt");
		
		/*disp("D:");
		disp(D);
		disp("Omega:");
		disp(Omega);*/
		
		// Run matrix completion
		MatrixCompletion matrixCompletion = new MatrixCompletion();
		matrixCompletion.feedData(D);
		matrixCompletion.feedIndices(Omega);
		tic();
		matrixCompletion.run();
		fprintf("Elapsed time: %.2f seconds.%n", toc());
		
		// Output
		Matrix A_hat = matrixCompletion.GetLowRankEstimation();
		
		fprintf("A*:\n");
		disp(A_star, 4);
		fprintf("A^:\n");
		disp(A_hat, 4);
		fprintf("D:\n");
		disp(D, 4);
		fprintf("rank(A*): %d\n", rank(A_star));
		fprintf("rank(A^): %d\n", rank(A_hat));
		fprintf("||A* - A^||_F: %.4f\n", norm(A_star.minus(A_hat), "fro"));
		
	}

	/**
	 * Observation real matrix.
	 */
	Matrix D;
	
	/**
	 * a sparse or dense logical matrix indicating the indices of samples
	 */
	Matrix Omega;
	
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
	 * Constructor.
	 */
	public MatrixCompletion() {
		
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
	 * Feed indices of samples.
	 * 
	 * @param Omega a sparse or dense logical matrix indicating the indices of samples
	 * 
	 */
	public void feedIndices(Matrix Omega) {
		this.Omega = Omega;
	}
	
	/**
	 * Feed indices of samples.
	 * 
	 * @param indices an {@code int} array for the indices of samples
	 * 
	 */
	public void feedIndices(int[] indices) {
		Omega = new SparseMatrix(size(D, 1), size(D, 2));
		linearIndexingAssignment(Omega, indices, 1);
	}
	
	/**
	 * Run matrix completion.
	 */
	public void run() {
		Matrix[] res = matrixCompletion(D, Omega);
		A = res[0];
		E = res[1];
	}
	
	/**
	 * Get the low-rank completed matrix.
	 * 
	 * @return the low-rank completed matrix
	 */
	public Matrix GetLowRankEstimation() {
		return A;
	}
	
	/**
	 * Get the error matrix between the original observation matrix and 
	 * its low-rank completed matrix.
	 * 
	 * @return error matrix
	 */
	public Matrix GetErrorMatrix() {
		return E;
	}

	/**
	 * Do matrix completion which solves the following convex 
	 * optimization problem:
	 * </br>
	 * min ||A||_*</br>
	 * s.t. D = A + E</br>
	 *      E(Omega) = 0</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values).</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param Omega a sparse or dense logical matrix indicating the indices of samples
	 * 
	 * @return a {@code Matrix} array [A, E] where A is the low-rank
	 * 		   completion from D, and E is the error matrix between A and D
	 * 
	 */
	static public Matrix[] matrixCompletion(Matrix D, Matrix Omega) {
		
		Matrix Y = zeros(size(D));
		Matrix E = zeros(size(D));
		Matrix A = minus(D, E);
		int m = size(D, 1);
		int n = size(D, 2);
		double mu = 1 / norm(D, 2);
		// Sampling density
		double rou_s = sumAll(gt(Omega, 0)) / (m * n);
		// The relation between rou and rou_s is obtained by regression
		double rou = 1.2172 + 1.8588 * rou_s;
		int k = 0;
		double norm_D = norm(D, "fro");
		double e1 = 1e-7;
		double e2 = 1e-6;
		double c1 = 0;
		double c2 = 0;
		double mu_old = 0;
		Matrix E_old = E.copy();
		Matrix[] SVD = null;
		
		while (true) {
			
			// Stopping criteria
			if (k > 1) {
				c1 = norm(D.minus(A).minus(E), "fro") / norm_D;
				c2 = mu_old * norm(E.minus(E_old), "fro") / norm_D;
				// fprintf("k = %d, c2: %.4f\n", k, c2);
				if (c1 <= e1 && c2 <= e2)
					break;
			}
			
			// E_old = E;
			assign(E_old, E);
			mu_old = mu;
		    
		    // A_{k+1} = argmin_A L(A, E_k, Y_k, mu_k)
			/*disp(plus(minus(D, E), rdivide(Y, mu)));
			Matrix SVDInput = plus(minus(D, E), rdivide(Y, mu));
			IO.saveMatrix(SVDInput, "SVDInput");*/
		    SVD = svd(plus(minus(D, E), rdivide(Y, mu)));
		    // disp(full(SVD[1]));
		    A = SVD[0].mtimes(shrinkage(SVD[1], 1 / mu)).mtimes(SVD[2].transpose());

		    // E_{k+1} = argmin_E L(A_{k+1}, E, Y_k, mu_k)
		    // E = D.minus(A);
		    minus(E, D, A);
		    logicalIndexingAssignment(E, Omega, 0);
		    
		    // Y = Y.plus(times(mu, D.minus(A).minus(E)));
		    plusAssign(Y, mu, D.minus(A).minus(E));
		    /*disp("Y:");
		    disp(Y);*/
		    
		    // disp(E);
		    /*if (Double.isNaN(sumAll(E))) {
		    	IO.saveMatrix(D, "D.txt");
		    	IO.saveMatrix(Omega, "Omega.txt");
		    	exit(1);
		    }*/
		    // Update mu_k to mu_{k+1}

		    if (norm(E.minus(E_old), "fro") * mu / norm_D < e2)
		    	mu = rou * mu;
		    
		    // fprintf("mu: %f%n", mu);

		    k = k + 1;

		}

		Matrix[] res = new Matrix[2];
		res[0] = A;
		res[1] = E;
		return res;
		
	}
	
	/**
	 * Do matrix completion which solves the following convex 
	 * optimization problem:
	 * </br>
	 * min ||A||_*</br>
	 * s.t. D = A + E</br>
	 *      E(Omega) = 0</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values).</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param indices an {@code int} array for the indices of samples
	 * 
	 * @return a {@code Matrix} array [A, E] where A is the low-rank
	 * 		   completion from D, and E is the error matrix between A and D
	 * 
	 */
	static public Matrix[] matrixCompletion(Matrix D, int[] indices) {
		Matrix Omega = new SparseMatrix(size(D, 1), size(D, 2));
		linearIndexingAssignment(Omega, indices, 1);
		return matrixCompletion(D, Omega);
	}
	
}
