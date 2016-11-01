package org.laml.ml.manifold;

import static org.laml.la.io.IO.loadMatrixFromDocTermCountFile;
import static org.laml.la.io.IO.saveMatrix;
import static org.laml.ml.kernel.KernelByColumns.calcKernel;
import static org.laml.la.utils.Matlab.denseMatrix2DenseColumnVectors;
import static org.laml.la.utils.Matlab.diag;
import static org.laml.la.utils.Matlab.dotDivide;
import static org.laml.la.utils.Matlab.eps;
import static org.laml.la.utils.Matlab.eye;
import static org.laml.la.utils.Matlab.find;
import static org.laml.la.utils.Matlab.l2DistanceByColumns;
import static org.laml.la.utils.Matlab.max;
import static org.laml.la.utils.Matlab.mrdivide;
import static org.laml.la.utils.Matlab.size;
import static org.laml.la.utils.Matlab.sort;
import static org.laml.la.utils.Matlab.sparseMatrix2SparseColumnVectors;
import static org.laml.la.utils.Matlab.sparseMatrix2SparseRowVectors;
import static org.laml.la.utils.Matlab.sparseRowVectors2SparseMatrix;
import static org.laml.la.utils.Matlab.speye;
import static org.laml.la.utils.Matlab.sqrt;
import static org.laml.la.utils.Matlab.sum;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Printer.*;
import static org.laml.la.utils.Time.*;
import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.vector.SparseVector;
import org.laml.la.vector.Vector;
import org.laml.ml.options.GraphOptions;
import org.laml.la.utils.FindResult;

/***
 * Java implementation of commonly used manifold learning functions.
 * 
 * @version 1.0 Jan. 27th, 2014
 * @author Mingjie Qian
 */
public class ManifoldByColumns {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		String filePath = "CNN - DocTermCount.txt";
		Matrix X = loadMatrixFromDocTermCountFile(filePath);
		int NSample = Math.min(20, X.getColumnDimension());
		X = X.getSubMatrix(0, X.getRowDimension() - 1, 0, NSample - 1);
		System.out.println(String.format("%d samples loaded", X.getColumnDimension()));
		GraphOptions options = new GraphOptions();
		options.graphType = "nn";
		String type = options.graphType;
		double NN = options.graphParam;
		System.out.println(String.format("Graph type: %s with NN: %d", type, (int)NN));
		
		// Parameter setting for text data
		options.kernelType = "cosine";
		options.graphDistanceFunction = "cosine";
		
		// Parameter setting for image data
		/*options.kernelType = "rbf";
		options.graphDistanceFunction = "euclidean";*/
		
		options.graphNormalize = true;
		options.graphWeightType = "heat";
		
		boolean show = true && !false;
		
		// Test adjacency function - pass
		tic();
		String DISTANCEFUNCTION = options.graphDistanceFunction;
		Matrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String adjacencyFilePath = "adjacency.txt";
		saveMatrix(adjacencyFilePath, A);
		if (show)
			disp(A.getSubMatrix(0, 9, 0, 9));
		
		// Test laplacian function - pass
		tic();
		Matrix L = laplacian(X, type, options);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String LaplacianFilePath = "Laplacian.txt";
		saveMatrix(LaplacianFilePath, L);
		if (show)
			disp(L.getSubMatrix(0, 9, 0, 9));
		
		// Test local learning regularization - pass
		NN = options.graphParam;
		String DISTFUNC = options.graphDistanceFunction;
		String KernelType = options.kernelType;
		double KernelParam = options.kernelParam;
		double lambda = 0.001;
		tic();
		Matrix LLR_text = calcLLR(X, NN, DISTFUNC, KernelType, KernelParam, lambda);
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		String LLRFilePath = "localLearningRegularization.txt";
		saveMatrix(LLRFilePath, LLR_text);
		if (show)
			display(LLR_text.getSubMatrix(0, 9, 0, 9));
		
	}
	
	/**
	 * Calculate the graph Laplacian of the adjacency graph of a data set
	 * represented as columns of a matrix X.
	 * 
	 * @param X data matrix with each column being a sample
	 * 
	 * @param type graph type, either "nn" or "epsballs"
	 * 
	 * @param options 
	 *        data structure containing the following fields
	 *        NN - integer if type is "nn" (number of nearest neighbors),
	 *             or size of "epsballs"
	 *        DISTANCEFUNCTION - distance function used to make the graph
	 *        WEIGHTTYPPE = "binary" | "distance" | "heat" | "inner"
	 * 	      WEIGHTPARAM = width for heat kernel
	 * 	      NORMALIZE = 0 | 1 whether to return normalized graph Laplacian or not
	 * 
	 * @return a sparse symmetric N x N matrix
	 * 
	 */
	public static Matrix laplacian(Matrix X, String type, GraphOptions options) {
		
		System.out.println("Computing Graph Laplacian...");
		
		double NN = options.graphParam;
		String DISTANCEFUNCTION = options.graphDistanceFunction;
		String WEIGHTTYPE = options.graphWeightType;
		double WEIGHTPARAM = options.graphWeightParam;
		boolean NORMALIZE = options.graphNormalize;
		
		if (WEIGHTTYPE.equals("inner") && !DISTANCEFUNCTION.equals("cosine"))
		    System.err.println("WEIGHTTYPE and DISTANCEFUNCTION mismatch.");
		
		// Calculate the adjacency matrix for DATA
		Matrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
		
		// W could be viewed as a similarity matrix
		Matrix W = A.copy();
		
		// Disassemble the sparse matrix
		FindResult findResult = find(A);
		int[] A_i = findResult.rows;
		int[] A_j = findResult.cols;
		double[] A_v = findResult.vals;
		
		if (WEIGHTTYPE.equals("distance")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], A_v[i]);
			}
		} else if (WEIGHTTYPE.equals("inner")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 1 - A_v[i] / 2);
			}
		} else if (WEIGHTTYPE.equals("binary")) {
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 1);
			}
		} else if (WEIGHTTYPE.equals("heat")) {
			double t = -2 * WEIGHTPARAM * WEIGHTPARAM;
			for (int i = 0; i < A_i.length; i++) {
				W.setEntry(A_i[i], A_j[i], 
			               Math.exp(A_v[i] * A_v[i] / t));
			}
		} else {
			System.err.println("Unknown Weight Type.");
		}
		
		Matrix D = null;
		Vector V = sum(W, 2);
		Matrix L = null;
		if (!NORMALIZE)
			L = diag(V).minus(W);
		else {
			// Normalized Laplacian
			D = diag(dotDivide(1, sqrt(V)));
			L = speye(size(W, 1)).minus(D.mtimes(W).mtimes(D));
		}
		
		return L;
		
	}
	
	/**
	 * Compute the symmetric adjacency matrix of the data set represented as
	 * a real data matrix X. The diagonal elements of the sparse symmetric
	 * adjacency matrix are all zero indicating that a sample should not be
	 * a neighbor of itself. Note that in some cases, neighbors of a sample
	 * may coincide with the sample itself, we set eps for those entries in
	 * the sparse symmetric adjacency matrix.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param type graph type, either "nn" or "epsballs" ("eps")
	 * 
	 * @param param integer if type is "nn", real number if type is "epsballs" ("eps")
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *                 to an M x N distance matrix (D: dimensionality)
     *                 either "euclidean" or "cosine"
     *                 
	 * @return a sparse symmetric N x N  matrix of distances between the
     *         adjacent points
     *         
	 */
	public static Matrix adjacency(Matrix X, String type, double param, String distFunc) {
		Matrix A = adjacencyDirected(X, type, param, distFunc);
		return max(A, A.transpose());
	}
	
	/**
	 * Compute the directed adjacency matrix of the data set represented as
	 * a real data matrix X. The diagonal elements of the sparse directed 
	 * adjacency matrix are all zero indicating that a sample should not be
	 * a neighbor of itself. Note that in some cases, neighbors of a sample
	 * may coincide with the sample itself, we set eps for those entries in
	 * the sparse directed adjacency matrix.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param type graph type, either "nn" or "epsballs" ("eps")
	 * 
	 * @param param integer if type is "nn", real number if type is "epsballs" ("eps")
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *                 to an M x N distance matrix (D: dimensionality)
     *                 either "euclidean" or "cosine"
     *                 
	 * @return a sparse N x N matrix of distances between the
     *         adjacent points, not necessarily symmetric
     *         
	 */
	public static Matrix adjacencyDirected(Matrix X, String type, double param, String distFunc) {
		
		System.out.println("Computing directed adjacency graph...");
		
		int n = size(X, 2);
		
		if (type.equals("nn")) {
			System.out.println(String.format("Creating the adjacency matrix. Nearest neighbors, N = %d.", (int)param));
		} else if (type.equals("epsballs") || type.equals("eps")) {
			System.out.println(String.format("Creating the adjacency matrix. Epsilon balls, eps = %f.", param));
		} else {
			System.err.println("type should be either \"nn\" or \"epsballs\" (\"eps\")");
			System.exit(1);
		}
		
		Matrix A = new SparseMatrix(n, n);
		
		Matrix dt = null;
		for (int i = 0; i < n; i++) {
			
			if (distFunc.equals("euclidean")) {
				dt = euclideanByColumns(X.getColumnMatrix(i), X);
			} else if (distFunc.equals("cosine")) {
				dt = cosineByColumns(X.getColumnMatrix(i), X);
			}
			
			Matrix[] sortResult = sort(dt, 2);
			Matrix Z = sortResult[0];
			double[][] IX = ((DenseMatrix) sortResult[1]).getData();
			
			if (type.equals("nn")) {
				for (int j = 0; j <= param; j++ ) {
					if ((int)IX[0][j] != i)
						A.setEntry(i, (int)IX[0][j], Z.getEntry(0, j) + eps);
				}
			} else if (type.equals("epsballs") || type.equals("eps")) {
				int j = 0;
				while (Z.getEntry(0, j) <= param) {
					if ((int)IX[0][j] != i)
						A.setEntry(i, (int)IX[0][j], Z.getEntry(0, j) + eps);
					j++;
				}
			}
			
		}
		
		return A;
		
	}
	
	/**
	 * Compute the cosine distance matrix between column vectors in matrix A
	 * and column vectors in matrix B.
	 * 
	 * @param A data matrix with each column being a feature vector
	 * 
	 * @param B data matrix with each column being a feature vector
	 * 
	 * @return an n_A X n_B matrix with its (i, j) entry being the cosine
	 * distance between i-th feature vector in A and j-th feature
	 * vector in B, i.e.,
	 * ||A(:, i) - B(:, j)|| = 1 - A(:, i)' * B(:, j) / || A(:, i) || * || B(:, j)||
	 */
	public static Matrix cosineByColumns(Matrix A, Matrix B) {
		
		/*Matrix AA = sum(times(A, A));
		Matrix BB = sum(times(B, B));
		Matrix AB = A.transpose().mtimes(B);
		Matrix C = times(scalarDivide(1, sqrt(kron(AA.transpose(), BB))), AB);
		return C.scalarMultiply(-1.0).scalarAdd(1.0);*/
		
		double[] AA = sum(times(A, A)).getPr();
		double[] BB = sum(times(B, B)).getPr();
		Matrix AB = A.transpose().mtimes(B);
		int M = AB.getRowDimension();
		int N = AB.getColumnDimension();
		double v = 0;
		if (AB instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) AB).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					v = resRow[j];
					resRow[j] = 1 - v / Math.sqrt(AA[i] * BB[j]);
				}
			}
		} else if (AB instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) AB).getPr();
			int[] ir = ((SparseMatrix) AB).getIr();
			int[] jc = ((SparseMatrix) AB).getJc();
			for (int j = 0; j < N; j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					pr[k] /= -Math.sqrt(AA[ir[k]] * BB[j]);
				}
			}
			AB = AB.plus(1);
		}
		
		return AB;
		
	}
	
	/**
	 * Compute the Euclidean distance matrix between column vectors in matrix A
	 * and column vectors in matrix B.
	 * 
	 * @param A data matrix with each column being a feature vector
	 * 
	 * @param B data matrix with each column being a feature vector
	 * 
	 * @return an n_A X n_B matrix with its (i, j) entry being Euclidean
	 * distance between i-th feature vector in A and j-th feature
	 * vector in B, i.e., || X(:, i) - Y(:, j) ||_2
	 * 
	 */
	public static Matrix euclideanByColumns(Matrix A, Matrix B) {
		return l2DistanceByColumns(A, B);
	}
	
	/**
	 * Compute local learning regularization matrix. Local learning
	 * regularization only depends on kernel selection, distance
	 * function, and neighborhood size.
	 * 
	 * @param X data matrix with each column being a feature vector
	 * 
	 * @param NN number of nearest neighbor
	 * 
	 * @param distFunc function mapping a (D x M) and a (D x N) matrix
     *        to an M x N distance matrix (D: dimensionality)
     *        either "euclidean" or "cosine"
     *        
	 * @param kernelType  'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam    --    | degree | sigma |    --
	 * 
	 * @param lambda graph regularization parameter
	 * 
	 * @return local learning regularization matrix
	 * 
	 */
	public static Matrix calcLLR(Matrix X,
			double NN, String distFunc, String kernelType,
			double kernelParam, double lambda) {
		
		String type = "nn";
		double param = NN;
		Matrix A = adjacencyDirected(X, type, param, distFunc);
		Matrix K = calcKernel(kernelType, kernelParam, X);
		
		int NSample = size(X, 2);
		// int NFeature = size(X, 1);
		int n_i = (int)param;
		Matrix I_i = eye(n_i);
		Matrix I = speye(NSample);
		
		Matrix G = A.copy();
		
		int[] neighborIndices_i = null;
		// Matrix neighborhood_X_i = null;
		Vector[] neighborhood_X_i = null;
		Matrix K_i = null;
		Matrix k_i = null;
		Vector x_i = null;
		Matrix alpha_i = null;
		// int[] IDs = colon(0, NFeature - 1);
		Vector[] Vs = sparseMatrix2SparseRowVectors(A);
		Vector[] Xs = null;
		if (X instanceof DenseMatrix) {
			Xs = denseMatrix2DenseColumnVectors(X);
		} else if (X instanceof SparseMatrix) {
			Xs = sparseMatrix2SparseColumnVectors(X);
		}
		Vector[] Gs = new Vector[NSample];
		for (int i = 0; i < NSample; i++) {
			// neighborIndices_i = find(A.getRowVector(i));
			neighborIndices_i = find(Vs[i]);
			// neighborhood_X_i = X.getSubMatrix(IDs, neighborIndices_i);
			neighborhood_X_i = new Vector[neighborIndices_i.length];
			for (int k = 0; k < neighborIndices_i.length; k++) {
				neighborhood_X_i[k] = Xs[neighborIndices_i[k]];
			}
			K_i = K.getSubMatrix(neighborIndices_i, neighborIndices_i);
			// x_i = X.getColumnMatrix(i);
			x_i = Xs[i];
			k_i = calcKernel(kernelType, kernelParam, new Vector[] {x_i}, neighborhood_X_i);
			alpha_i = mrdivide(k_i, I_i.times(n_i * lambda).plus(K_i));
			// setSubMatrix(G, new int[]{i}, neighborIndices_i, alpha_i);
			Gs[i] = new SparseVector(neighborIndices_i, ((DenseMatrix) alpha_i).getData()[0], neighborIndices_i.length, NSample);
		}
		G = sparseRowVectors2SparseMatrix(Gs);
		Matrix T = G.minus(I);
		Matrix L = T.transpose().mtimes(T);
		
		return L;
		
	}

}
