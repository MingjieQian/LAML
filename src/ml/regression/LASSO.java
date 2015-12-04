package ml.regression;

import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.or;
import static ml.utils.InPlaceOperator.plus;
import static ml.utils.InPlaceOperator.plusAssign;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Matlab.abs;
import static ml.utils.Matlab.eye;
import static ml.utils.Matlab.gt;
import static ml.utils.Matlab.horzcat;
import static ml.utils.Matlab.inf;
import static ml.utils.Matlab.logicalIndexingAssignment;
import static ml.utils.Matlab.lt;
import static ml.utils.Matlab.minus;
import static ml.utils.Matlab.mldivide;
import static ml.utils.Matlab.mtimes;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.not;
import static ml.utils.Matlab.plus;
import static ml.utils.Matlab.pow;
import static ml.utils.Matlab.size;
import static ml.utils.Matlab.subplus;
import static ml.utils.Matlab.sum;
import static ml.utils.Matlab.sumAll;
import static ml.utils.Matlab.times;
import static ml.utils.Matlab.uminus;
import static ml.utils.Matlab.vertcat;
import static ml.utils.Matlab.zeros;
import static ml.utils.Printer.display;
import static ml.utils.Printer.fprintf;
import static ml.utils.Time.tic;
import static ml.utils.Time.toc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.DenseVector;
import la.vector.Vector;
import ml.options.Options;
import ml.utils.Matlab;

/***
 * A Java implementation of LASSO, which solves the following
 * convex optimization problem:
 * </p>
 * min_W 2\1 || Y - X * W ||_F^2 + lambda * || W ||_1</br>
 * where X is an n-by-p data matrix with each row bing a p
 * dimensional data vector and Y is an n-by-ny dependent
 * variable matrix.
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 1st, 2014
 */
public class LASSO extends Regression {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {{1, 2, 3, 2},
						   {4, 2, 3, 6},
						   {5, 1, 4, 1}}; 
		
		double[][] depVars = {{3, 2},
							  {2, 3},
							  {1, 4}};

		Options options = new Options();
		options.maxIter = 600;
		options.lambda = 0.1;
		options.verbose = !true;
		options.calc_OV = !true;
		options.epsilon = 1e-5;
		
		Regression LASSO = new LASSO(options);
		LASSO.feedData(data);
		LASSO.feedDependentVariables(depVars);
		
		tic();
		LASSO.train();
		fprintf("Elapsed time: %.3f seconds\n\n", toc());
		
		fprintf("Projection matrix:\n");
		display(LASSO.W);
		
		Matrix Yt = LASSO.predict(data);
		fprintf("Predicted dependent variables:\n");
		display(Yt);
		
	}

	/**
	 * Regularization parameter.
	 */
	private double lambda;
	
	/**
	 * If compute objective function values during
	 * the iterations or not.
	 */
	private boolean calc_OV;
	
	/**
	 * If show computation detail during iterations or not.
	 */
	private boolean verbose;

	public LASSO() {
		super();
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}

	public LASSO(double epsilon) {
		super(epsilon);
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(int maxIter, double epsilon) {
		super(maxIter, epsilon);
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(double lambda, int maxIter, double epsilon) {
		super(maxIter, epsilon);
		this.lambda = lambda;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(Options options) {
		super(options);
		lambda = options.lambda;
		calc_OV = options.calc_OV;
		verbose = options.verbose;
	}

	@Override
	public void train() {
		W = train(X, Y);
	}

	@Override
	public void loadModel(String filePath) {
		
		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public void saveModel(String filePath) {
		
		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static Matrix train(Matrix X, Matrix Y, Options options) {

		int p = size(X, 2);
		int ny = size(Y, 2);
		double epsilon = options.epsilon;
		int maxIter = options.maxIter;
		double lambda = options.lambda;
		boolean calc_OV = options.calc_OV;
		boolean verbose = options.verbose;
		
		/*XNX = [X, -X];
		H_G = XNX' * XNX;
		D = repmat(diag(H_G), [1, n_y]);
		XNXTY = XNX' * Y;
	    A = (X' * X + lambda  * eye(p)) \ (X' * Y);*/
		
		Matrix XNX = horzcat(X, uminus(X));
		Matrix H_G = XNX.transpose().mtimes(XNX);
		// Matrix D = repmat(diag(H_G), 1, ny);
		double[] Q = new double[size(H_G, 1)];
		for (int i = 0; i < Q.length; i++) {
			Q[i] = H_G.getEntry(i, i);
		}
		Matrix XNXTY = XNX.transpose().mtimes(Y);
		Matrix A = mldivide(
						   plus(X.transpose().mtimes(X), times(lambda, eye(p))), 
						   X.transpose().mtimes(Y)
						   );
		
		/*AA = [subplus(A); subplus(-A)];
		C = -XNXTY + lambda;
		Grad = C + H_G * AA;
		tol = epsilon * norm(Grad);
		PGrad = zeros(size(Grad));*/
		
		Matrix AA = vertcat(subplus(A), subplus(uminus(A)));
		Matrix C = plus(uminus(XNXTY), lambda);
		Matrix Grad = plus(C, mtimes(H_G, AA));
		double tol = epsilon * norm(Grad);
		Matrix PGrad = zeros(size(Grad));
		
		ArrayList<Double> J = new ArrayList<Double>();
		double fval = 0;
		// J(1) = sum(sum((Y - X * A).^2)) / 2 + lambda * sum(sum(abs(A)));
		if (calc_OV) {
			fval = sum(sum(pow(minus(Y, mtimes(X, A)), 2))) / 2 +
				   lambda * sum(sum(abs(A)));
			J.add(fval);
		}
		
		Matrix I_k = Grad.copy();
		Matrix I_k_com = null;
		double d = 0;
		// Matrix tmp = null;
		int k = 0;
		
		Vector SFPlusCi = null;
		Matrix S = H_G;
		Vector[] SRows = null;
		if (H_G instanceof DenseMatrix)
			SRows = Matlab.denseMatrix2DenseRowVectors(S);
		else
			SRows = Matlab.sparseMatrix2SparseRowVectors(S);
		
		Vector[] CRows = null;
		if (C instanceof DenseMatrix)
			CRows = Matlab.denseMatrix2DenseRowVectors(C);
		else
			CRows = Matlab.sparseMatrix2SparseRowVectors(C);
		
		double[][] FData = ((DenseMatrix) AA).getData();
		double[] FRow = null;
		double[] pr = null;
		int K = 2 * p;
		
		while (true) {	
			
			/*I_k = Grad < 0 | AA > 0;
		    I_k_com = not(I_k);
		    PGrad(I_k) = Grad(I_k);
		    PGrad(I_k_com) = 0;*/
			
			or(I_k, lt(Grad, 0), gt(AA, 0));
			I_k_com = not(I_k);
			assign(PGrad, Grad);
			logicalIndexingAssignment(PGrad, I_k_com, 0);
			
			d = norm(PGrad, inf);
		    if (d < tol) {
		        if (verbose)
		            System.out.println("Converge successfully!");
		        break;
		    }
		    
		    /*for i = 1:2*p
		            AA(i, :) = max(AA(i, :) - (C(i, :) + H_G(i, :) * AA) ./ (D(i, :)), 0);
		    end
		    A = AA(1:p,:) - AA(p+1:end,:);*/
		    
		    /*for (int i = 0; i < 2 * p; i++) {
		    	tmp = max(
		    			minus(
	    					AA.getRowMatrix(i), 
	    					rdivide(
    							plus(C.getRowMatrix(i), mtimes(H_G.getRowMatrix(i), AA)),
    							Q[i]
    						    )
		    				),
		    		    0.0
		    		    );
		    						    
		    	AA.setRowMatrix(i, tmp);
		    }
		    
		    Grad = plus(C, mtimes(H_G, AA));*/
		    
		    for (int i = 0; i < K; i++) {
				SFPlusCi = SRows[i].operate(AA);
				plusAssign(SFPlusCi, CRows[i]);
				timesAssign(SFPlusCi, 1 / Q[i]);
				pr = ((DenseVector) SFPlusCi).getPr();
				// F(i, :) = max(F(i, :) - (S(i, :) * F + C(i, :)) / D[i]), 0);
				// F(i, :) = max(F(i, :) - SFPlusCi, 0)
				FRow = FData[i];
				for (int j = 0; j < AA.getColumnDimension(); j++) {
					FRow[j] = Math.max(FRow[j] - pr[j], 0);
				}	
			}
		    
		    // Grad = plus(C, mtimes(H_G, AA));
		    plus(Grad, C, mtimes(H_G, AA));
		    
		    k = k + 1;
		    if (k > maxIter) {
		    	if (verbose)
		    		System.out.println("Maximal iterations");
		    	break;
		    }
		    
		    if (calc_OV) {
				fval = sum(sum(pow(minus(Y, mtimes(XNX, AA)), 2))) / 2 +
						lambda * sum(sum(abs(AA)));
				J.add(fval);
			}
		    
		    if (k % 10 == 0 && verbose) {
		    	if (calc_OV)
		    		System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
		    	else
		    		System.out.format("Iter %d - ||PGrad||: %f\n", k, d);

		    }
			
		}
		
		A = minus(
	    		  AA.getSubMatrix(0, p - 1, 0, ny - 1),
	    		  AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1)
	    		  );
		return A;
		
	}
	
	@Override
	public Matrix train(Matrix X, Matrix Y) {
		
		int p = size(X, 2);
		int ny = size(Y, 2);
		
		/*XNX = [X, -X];
		H_G = XNX' * XNX;
		D = repmat(diag(H_G), [1, n_y]);
		XNXTY = XNX' * Y;
	    A = (X' * X + lambda  * eye(p)) \ (X' * Y);*/
		
		Matrix XNX = horzcat(X, uminus(X));
		Matrix H_G = XNX.transpose().mtimes(XNX);
		// Matrix D = repmat(diag(H_G), 1, ny);
		double[] Q = new double[size(H_G, 1)];
		for (int i = 0; i < Q.length; i++) {
			Q[i] = H_G.getEntry(i, i);
		}
		Matrix XNXTY = XNX.transpose().mtimes(Y);
		Matrix A = mldivide(
						   plus(X.transpose().mtimes(X), times(lambda, eye(p))), 
						   X.transpose().mtimes(Y)
						   );
		
		/*AA = [subplus(A); subplus(-A)];
		C = -XNXTY + lambda;
		Grad = C + H_G * AA;
		tol = epsilon * norm(Grad);
		PGrad = zeros(size(Grad));*/
		
		Matrix AA = vertcat(subplus(A), subplus(uminus(A)));
		Matrix C = plus(uminus(XNXTY), lambda);
		Matrix Grad = plus(C, mtimes(H_G, AA));
		double tol = epsilon * norm(Grad);
		Matrix PGrad = zeros(size(Grad));
		
		ArrayList<Double> J = new ArrayList<Double>();
		double fval = 0;
		// J(1) = sum(sum((Y - X * A).^2)) / 2 + lambda * sum(sum(abs(A)));
		if (calc_OV) {
			fval = sumAll(pow(Matlab.minus(Y, mtimes(X, A)), 2)) / 2 +
				   lambda * sum(sum(abs(A)));
			J.add(fval);
		}
		
		Matrix I_k = Grad.copy();
		Matrix I_k_com = null;
		double d = 0;
		// Matrix tmp = null;
		int k = 0;
		
		Vector SFPlusCi = null;
		Matrix S = H_G;
		Vector[] SRows = null;
		if (H_G instanceof DenseMatrix)
			SRows = Matlab.denseMatrix2DenseRowVectors(S);
		else
			SRows = Matlab.sparseMatrix2SparseRowVectors(S);
		
		Vector[] CRows = null;
		if (C instanceof DenseMatrix)
			CRows = Matlab.denseMatrix2DenseRowVectors(C);
		else
			CRows = Matlab.sparseMatrix2SparseRowVectors(C);
		
		double[][] FData = ((DenseMatrix) AA).getData();
		double[] FRow = null;
		double[] pr = null;
		int K = 2 * p;
		
		while (true) {	
			
			/*I_k = Grad < 0 | AA > 0;
		    I_k_com = not(I_k);
		    PGrad(I_k) = Grad(I_k);
		    PGrad(I_k_com) = 0;*/
			
			or(I_k, lt(Grad, 0), gt(AA, 0));
			I_k_com = not(I_k);
			assign(PGrad, Grad);
			logicalIndexingAssignment(PGrad, I_k_com, 0);
			
			d = norm(PGrad, inf);
		    if (d < tol) {
		        if (verbose)
		            System.out.println("Converge successfully!");
		        break;
		    }
		    
		    /*for i = 1:2*p
		            AA(i, :) = max(AA(i, :) - (C(i, :) + H_G(i, :) * AA) ./ (D(i, :)), 0);
		    end
		    A = AA(1:p,:) - AA(p+1:end,:);*/
		    
		    /*for (int i = 0; i < 2 * p; i++) {
		    	tmp = max(
		    			minus(
	    					AA.getRowMatrix(i),
	    					rdivide(
    							plus(C.getRowMatrix(i), mtimes(H_G.getRowMatrix(i), AA)),
    							Q[i]
    						    )
		    				),
		    		    0.0
		    		    );
		    						    
		    	AA.setRowMatrix(i, tmp);
		    }*/
		    
		    for (int i = 0; i < K; i++) {
				SFPlusCi = SRows[i].operate(AA);
				plusAssign(SFPlusCi, CRows[i]);
				timesAssign(SFPlusCi, 1 / Q[i]);
				pr = ((DenseVector) SFPlusCi).getPr();
				// F(i, :) = max(F(i, :) - (S(i, :) * F + C(i, :)) / D[i]), 0);
				// F(i, :) = max(F(i, :) - SFPlusCi, 0)
				FRow = FData[i];
				for (int j = 0; j < AA.getColumnDimension(); j++) {
					FRow[j] = Math.max(FRow[j] - pr[j], 0);
				}	
			}
		    
		    // Grad = plus(C, mtimes(H_G, AA));
		    plus(Grad, C, mtimes(H_G, AA));
		    
		    k = k + 1;
		    if (k > maxIter) {
		    	if (verbose)
		    		System.out.println("Maximal iterations");
		    	break;
		    }
		    
		    if (calc_OV) {
				fval = sum(sum(pow(minus(Y, mtimes(XNX, AA)), 2))) / 2 +
						lambda * sum(sum(abs(AA)));
				J.add(fval);
			}
		    
		    if (k % 10 == 0 && verbose) {
		    	if (calc_OV)
		    		System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
		    	else
		    		System.out.format("Iter %d - ||PGrad||: %f\n", k, d);

		    }
			
		}
		
		A = minus(
	    		  AA.getSubMatrix(0, p - 1, 0, ny - 1),
	    		  AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1)
	    		  );
		return A;
		
	}

	@Override
	public Matrix train(Matrix X, Matrix Y, Matrix W0) {
		p = W0.getRowDimension();
		ny = W0.getColumnDimension();
		// n = X.getRowDimension();
		/*XNX = [X, -X];
		H_G = XNX' * XNX;
		D = repmat(diag(H_G), [1, n_y]);
		XNXTY = XNX' * Y;
	    A = (X' * X + lambda  * eye(p)) \ (X' * Y);*/
		
		Matrix XNX = horzcat(X, uminus(X));
		Matrix H_G = XNX.transpose().mtimes(XNX);
		// Matrix D = repmat(diag(H_G), 1, ny);
		double[] Q = new double[size(H_G, 1)];
		for (int i = 0; i < Q.length; i++) {
			Q[i] = H_G.getEntry(i, i);
		}
		Matrix XNXTY = XNX.transpose().mtimes(Y);
		/*Matrix A = mldivide(
						   plus(X.transpose().mtimes(X), times(lambda, eye(p))), 
						   X.transpose().mtimes(Y)
						   );*/
		Matrix A = W0.copy();
		
		/*AA = [subplus(A); subplus(-A)];
		C = -XNXTY + lambda;
		Grad = C + H_G * AA;
		tol = epsilon * norm(Grad);
		PGrad = zeros(size(Grad));*/
		
		Matrix AA = vertcat(subplus(A), subplus(uminus(A)));
		Matrix C = plus(uminus(XNXTY), lambda);
		Matrix Grad = plus(C, mtimes(H_G, AA));
		double tol = epsilon * norm(Grad);
		Matrix PGrad = zeros(size(Grad));
		
		ArrayList<Double> J = new ArrayList<Double>();
		double fval = 0;
		// J(1) = sum(sum((Y - X * A).^2)) / 2 + lambda * sum(sum(abs(A)));
		if (calc_OV) {
			fval = sumAll(pow(Matlab.minus(Y, mtimes(X, A)), 2)) / 2 +
				   lambda * sum(sum(abs(A)));
			J.add(fval);
		}
		
		Matrix I_k = Grad.copy();
		Matrix I_k_com = null;
		double d = 0;
		// Matrix tmp = null;
		int k = 0;
		
		Vector SFPlusCi = null;
		Matrix S = H_G;
		Vector[] SRows = null;
		if (H_G instanceof DenseMatrix)
			SRows = Matlab.denseMatrix2DenseRowVectors(S);
		else
			SRows = Matlab.sparseMatrix2SparseRowVectors(S);
		
		Vector[] CRows = null;
		if (C instanceof DenseMatrix)
			CRows = Matlab.denseMatrix2DenseRowVectors(C);
		else
			CRows = Matlab.sparseMatrix2SparseRowVectors(C);
		
		double[][] FData = ((DenseMatrix) AA).getData();
		double[] FRow = null;
		double[] pr = null;
		int K = 2 * p;
		
		while (true) {	
			
			/*I_k = Grad < 0 | AA > 0;
		    I_k_com = not(I_k);
		    PGrad(I_k) = Grad(I_k);
		    PGrad(I_k_com) = 0;*/
			
			or(I_k, lt(Grad, 0), gt(AA, 0));
			I_k_com = not(I_k);
			assign(PGrad, Grad);
			logicalIndexingAssignment(PGrad, I_k_com, 0);
			
			d = norm(PGrad, inf);
		    if (d < tol) {
		        if (verbose)
		            System.out.println("LASSO converges successfully!");
		        break;
		    }
		    
		    /*for i = 1:2*p
		            AA(i, :) = max(AA(i, :) - (C(i, :) + H_G(i, :) * AA) ./ (D(i, :)), 0);
		    end
		    A = AA(1:p,:) - AA(p+1:end,:);*/
		    
		    /*for (int i = 0; i < 2 * p; i++) {
		    	tmp = max(
		    			minus(
	    					AA.getRowMatrix(i),
	    					rdivide(
    							plus(C.getRowMatrix(i), mtimes(H_G.getRowMatrix(i), AA)),
    							Q[i]
    						    )
		    				),
		    		    0.0
		    		    );
		    						    
		    	AA.setRowMatrix(i, tmp);
		    }*/
		    
		    for (int i = 0; i < K; i++) {
				SFPlusCi = SRows[i].operate(AA);
				plusAssign(SFPlusCi, CRows[i]);
				timesAssign(SFPlusCi, 1 / Q[i]);
				pr = ((DenseVector) SFPlusCi).getPr();
				// F(i, :) = max(F(i, :) - (S(i, :) * F + C(i, :)) / D[i]), 0);
				// F(i, :) = max(F(i, :) - SFPlusCi, 0)
				FRow = FData[i];
				for (int j = 0; j < AA.getColumnDimension(); j++) {
					FRow[j] = Math.max(FRow[j] - pr[j], 0);
				}	
			}
		    
		    // Grad = plus(C, mtimes(H_G, AA));
		    plus(Grad, C, mtimes(H_G, AA));
		    
		    k = k + 1;
		    if (k > maxIter) {
		    	if (verbose)
		    		System.out.println("Maximal iterations");
		    	break;
		    }
		    
		    if (calc_OV) {
				fval = sum(sum(pow(minus(Y, mtimes(XNX, AA)), 2))) / 2 +
						lambda * sum(sum(abs(AA)));
				J.add(fval);
			}
		    
		    if (k % 10 == 0 && verbose) {
		    	if (calc_OV)
		    		System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
		    	else
		    		System.out.format("Iter %d - ||PGrad||: %f\n", k, d);

		    }
			
		}
		
		A = minus(
	    		  AA.getSubMatrix(0, p - 1, 0, ny - 1),
	    		  AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1)
	    		  );
		return A;
	}

	@Override
	public void train(Matrix W0) {
		W = train(X, Y, W0);
	}

}