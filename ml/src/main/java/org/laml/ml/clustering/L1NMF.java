package ml.clustering;

import static la.io.IO.loadMatrixFromDocTermCountFile;
import static la.io.IO.saveMatrix;
import static la.utils.InPlaceOperator.assign;
import static la.utils.InPlaceOperator.operate;
import static la.utils.InPlaceOperator.plusAssign;
import static la.utils.InPlaceOperator.timesAssign;
import static la.utils.Matlab.abs;
import static la.utils.Matlab.diag;
import static la.utils.Matlab.find;
import static la.utils.Matlab.full;
import static la.utils.Matlab.isnan;
import static la.utils.Matlab.mldivide;
import static la.utils.Matlab.norm;
import static la.utils.Matlab.ones;
import static la.utils.Matlab.size;
import static la.utils.Matlab.subplus;
import static la.utils.Matlab.sumAll;
import static la.utils.Time.tic;
import static la.utils.Time.toc;

import java.util.ArrayList;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.DenseVector;
import la.vector.Vector;
import ml.options.KMeansOptions;
import ml.options.L1NMFOptions;
import ml.options.Options;
import la.utils.Matlab;

/***
 * A Java implementation for L1NMF which solves the following
 * optimization problem:
 * <p>
 * min || X - G * F ||_F^2 + gamma * ||F||_{sav} + nu * ||G||_{sav}</br>
 * s.t. G >= 0, F >= 0
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 31st, 2014
 */
public class L1NMF extends Clustering {

	public double epsilon;
	public int maxIter;

	public double gamma;
	public double mu;

	public boolean calc_OV;
	public boolean verbose;
	
	public ArrayList<Double> valueList;
	
	Matrix initializer = null;

	public L1NMF(Options options) {
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		gamma = options.gamma;
		mu = options.mu;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		nClus = options.nClus;
	}
	
	public L1NMF(L1NMFOptions L1NMFOptions) {
		maxIter = L1NMFOptions.maxIter;
		epsilon = L1NMFOptions.epsilon;
		gamma = L1NMFOptions.gamma;
		mu = L1NMFOptions.mu;
		verbose = L1NMFOptions.verbose;
		calc_OV = L1NMFOptions.calc_OV;
		nClus = L1NMFOptions.nClus;
	}

	public L1NMF() {
		L1NMFOptions options = new L1NMFOptions();
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		gamma = options.gamma;
		mu = options.mu;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		nClus = options.nClus;
	}
	
	public void initialize(Matrix G0) {
		
		if (G0 != null) {
			initializer = G0;
			return;
		}
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = nClus;
		kMeansOptions.maxIter = 50;
		kMeansOptions.verbose = true;
		
		System.out.println("Using KMeans to initialize...");
		Clustering KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(dataMatrix);
		// KMeans.initialize(null);
		KMeans.clustering();
		
		initializer = KMeans.getIndicatorMatrix();
		
	}
	
	@Override
	public void clustering() {
		if (initializer == null) {
			initialize(null);
			// initializer = indicatorMatrix;
		}
		clustering(initializer);
	}
	
	public void clustering(Matrix G0) {
		
		if (G0 == null) {
			initialize(null);
			G0 = initializer;
		}
		
		Matrix X = dataMatrix;
		Matrix G = G0;
		// Matrix F = X.mtimes(G).mtimes(new LUDecompositionImpl(G.transpose().mtimes(G)).getSolver().getInverse());
		Matrix F = mldivide(G.transpose().mtimes(G), G.transpose().mtimes(X));
		G = full(G);
			
		ArrayList<Double> J = new ArrayList<Double>();
		Matrix F_pos = subplus(F);
		F = F_pos.plus(0.2 * sumAll(F_pos) / find(F_pos).rows.length);
		
		Matrix E_F = ones(size(F)).times(gamma / 2);
		Matrix E_G = ones(size(G)).times( mu / 2 );

		if (calc_OV) {
			J.add(f(X, F, G, E_F, E_G));
		}

		int ind = 0;
		Matrix G_old = new DenseMatrix(size(G));
		double d = 0;

		while (true) {
			
			// G_old.setSubMatrix(G.getData(), 0, 0);
			assign(G_old, G);
			
			// Fixing F, updating G
			G = UpdateG(X, F, mu, G);

			// Fixing G, updating F
			F = UpdateF(X, G, gamma, F);


			ind = ind + 1;
			if (ind > maxIter) {
				System.out.println("Maximal iterations");
				break;
			}

			d = norm( G.minus(G_old), "fro");

			if (calc_OV) {
				J.add(f(X, F, G, E_F, E_G));
			}

			if (ind % 10 == 0 && verbose) {
				if (calc_OV) {
					System.out.println(String.format("Iteration %d, delta G: %f, J: %f", ind, d, J.get(J.size() - 1)));
					// System.out.flush();
				} else {
					System.out.println(String.format("Iteration %d, delta G: %f", ind, d));
					// System.out.flush();
				}
			}

			if (calc_OV) {
				if (Math.abs(J.get(J.size() - 2) - J.get(J.size() - 1)) < epsilon && d < epsilon) {
					System.out.println("Converge successfully!");
					break;	
				}
			} else if (d < epsilon) {
				System.out.println("Converge successfully!");
				break;
			}

			if (sumAll(isnan(G)) > 0) {
				break;
			}

		}
		
		centers = F;
		indicatorMatrix = G;
		valueList = J;

	}

	private Matrix UpdateG(Matrix X, Matrix F, double mu, Matrix G0) {

		// min|| X - G * F ||_F^2 + mu * || G ||_1
		// s.t. G >= 0

		int MaxIter = 10000;
		double epsilon = 1e-1;

		int K = Matlab.size(F, 1);
		int NExample = Matlab.size(X, 1);

		Matrix S = F.mtimes(F.transpose());
		Matrix C = X.mtimes(F.transpose());
		timesAssign(C, -1);
		plusAssign(C, mu / 2);

		double[] D = ((DenseVector) diag(S).getColumnVector(0)).getPr();
		
		Matrix G = G0;
		int ind = 0;
		double d = 0;

		Matrix G_old = new DenseMatrix(size(G));
		Vector GSPlusCj = new DenseVector(NExample);
		Vector[] SColumns = Matlab.denseMatrix2DenseColumnVectors(S);
		Vector[] CColumns = Matlab.denseMatrix2DenseColumnVectors(C);
		double[][] GData = ((DenseMatrix) G).getData();
		double[] pr = null;
		while (true) {

			assign(G_old, G);
			
			for (int j = 0; j < K; j++) {
				operate(GSPlusCj, G, SColumns[j]);
				// GSPlusCj = G.operate(SColumns[j]);
				plusAssign(GSPlusCj, CColumns[j]);
				timesAssign(GSPlusCj, 1 / D[j]);
				pr = ((DenseVector) GSPlusCj).getPr();
				// G(:, j) = max(G(:, j) - (G * S(:, j) + C(:, j)) / D[j]), 0);
				// G(:, j) = max(G(:, j) - GSPlusC_j, 0)
				for (int i = 0; i < NExample; i++) {
					GData[i][j] = Math.max(GData[i][j] - pr[i], 0);
				}
			}

			ind = ind + 1;
			if (ind > MaxIter) {
				break;
			}

			d = sumAll(abs(G.minus(G_old)));
			if (d < epsilon) {
				break;
			}

		}

		return G;

	}

	private Matrix UpdateF(Matrix X, Matrix G, double gamma, Matrix F0) {

		// min|| X - G * F ||_F^2 + gamma * || F ||_1
		// s.t. F >= 0

		int MaxIter = 10000;
		double epsilon = 1e-1;

		int K = Matlab.size(G, 2);
		int NFea = Matlab.size(X, 2);

		Matrix S = G.transpose().mtimes(G);
		Matrix C = G.transpose().mtimes(X);
		timesAssign(C, -1);
		plusAssign(C, gamma / 2);
		
		double[] D = ((DenseVector) diag(S).getColumnVector(0)).getPr();
		
		Matrix F = F0;
		int ind = 0;
		double d = 0;
		Matrix F_old = new DenseMatrix(size(F));
		Vector SFPlusCi = new DenseVector(NFea);
		Vector[] SRows = Matlab.denseMatrix2DenseRowVectors(S);
		Vector[] CRows = Matlab.denseMatrix2DenseRowVectors(C);
		double[][] FData = ((DenseMatrix) F).getData();
		double[] FRow = null;
		double[] pr = null;
		while (true) {

			assign(F_old, F);

			for (int i = 0; i < K; i++) {
				operate(SFPlusCi, SRows[i], F);
				// SFPlusCi = SRows[i].operate(F);
				plusAssign(SFPlusCi, CRows[i]);
				timesAssign(SFPlusCi, 1 / D[i]);
				pr = ((DenseVector) SFPlusCi).getPr();
				// F(i, :) = max(F(i, :) - (S(i, :) * F + C(i, :)) / D[i]), 0);
				// F(i, :) = max(F(i, :) - SFPlusCi, 0)
				FRow = FData[i];
				for (int j = 0; j < NFea; j++) {
					FRow[j] = Math.max(FRow[j] - pr[j], 0);
				}
			}

			ind = ind + 1;
			if (ind > MaxIter) {
				break;
			}

			d = sumAll(abs(F.minus(F_old)));
			if (d < epsilon) {
				break;
			}

		}

		return F;

	}

	private double f(Matrix X, Matrix F, Matrix G, Matrix E_F, Matrix E_G) {
		double fval = Math.pow(norm(X.minus(G.mtimes(F)), "fro"), 2) +
				2 * sumAll(E_F.times(F)) +
				2 * sumAll(E_G.times(G));
		/*return Math.pow(Matlab.norm( X.minus(F.mtimes(G.transpose())), "fro"), 2)
		+ 2 * Matlab.trace( E_F.transpose().mtimes(F) )
		+ 2 * Matlab.trace( E_G.transpose().mtimes(G) );*/
		return fval;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		String dataMatrixFilePath = "CNN - DocTermCount.txt";
		
		tic();
		Matrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		X = Matlab.getTFIDF(X);
		X = Matlab.normalizeByColumns(X);
		X = X.transpose();
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = 10;
		kMeansOptions.maxIter = 50;
		kMeansOptions.verbose = true;
		
		KMeans KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(X);
		// KMeans.initialize(null);
		KMeans.clustering();
		
		Matrix G0 = KMeans.getIndicatorMatrix();
		
		// Matrix X = Data.loadSparseMatrix("X.txt");
		// G0 = loadDenseMatrix("G0.txt");
		L1NMFOptions L1NMFOptions = new L1NMFOptions();
		L1NMFOptions.nClus = 10;
		L1NMFOptions.gamma = 1 * 0.0001;
		L1NMFOptions.mu = 1 * 0.1;
		L1NMFOptions.maxIter = 50;
		L1NMFOptions.verbose = true;
		L1NMFOptions.calc_OV = !true;
		L1NMFOptions.epsilon = 1e-5;
		Clustering L1NMF = new L1NMF(L1NMFOptions);
		L1NMF.feedData(X);
		// L1NMF.initialize(G0);
		
		// Matlab takes 12.5413 seconds
		// jblas takes 29.368 seconds
		// Commons-math takes 129 seconds (Using Array2DRowRealMatrix)
		// Commons-math takes 115 seconds (Using DenseMatrix)
		// start = System.currentTimeMillis();
		
		L1NMF.clustering(G0); // Use null for random initialization
		
		System.out.format("Elapsed time: %.3f seconds\n", toc());
		
		saveMatrix("F.txt", L1NMF.centers);
		saveMatrix("G.txt", L1NMF.indicatorMatrix);
		
	}

}
