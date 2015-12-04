package ml.sequence;

import static ml.utils.Printer.*;
import static java.lang.Math.log;
import static ml.utils.ArrayOperator.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;


/***
 * A Java implementation of basic Hidden Markov Model.
 * Observation symbols and state symbols are discrete
 * integers starting from 0. It is the responsibility
 * of users to maintain the mapping from IDs to
 * observation symbols and the mapping from IDs to
 * state symbols.
 * 
 * @author Mingjie Qian
 * @version 1.0, Dec. 23rd, 2013
 */
public class HMM {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int numStates = 3;
		int numObservations = 2;
		double epsilon = 1e-8;
		int maxIter = 10;

		double[] pi = new double[] {0.33, 0.33, 0.34};

		double[][] A = new double[][] {
				{0.5, 0.3, 0.2},
				{0.3, 0.5, 0.2},
				{0.2, 0.4, 0.4}
		};

		double[][] B = new double[][] {
				{0.7, 0.3},
				{0.5, 0.5},
				{0.4, 0.6}
		};

		// Generate the data sequences for training
		int D = 10000;
		int T_min = 5;
		int T_max = 10;
		int[][][] data = HMM.generateDataSequences(D, T_min, T_max, pi, A, B);
		int[][] Os = data[0];
		int[][] Qs = data[1];

		/*int[][] Os = new int[][] {
				{0, 0, 1, 0, 1},
				{0, 1, 1, 0, 1},
				{1, 0, 0, 1, 0},
				{0, 1, 1, 1, 0},
				{0, 1, 1, 0},
				{0, 1, 0, 1, 1}
				};

		int[][] Qs = new int[][] {
				{0, 0, 1, 0, 1},
				{0, 1, 1, 0, 1},
				{1, 0, 0, 0, 0},
				{0, 0, 1, 1, 1},
				{1, 1, 1, 0},
				{0, 1, 1, 1, 1}
				};*/

		boolean trainHMM = !false;
		if (trainHMM){
			HMM HMM = new HMM(numStates, numObservations, epsilon, maxIter);
			HMM.feedData(Os);
			HMM.feedLabels(Qs);
			HMM.train();

			fprintf("True Model Parameters: \n");
			fprintf("Initial State Distribution: \n");
			display(pi);
			fprintf("State Transition Probability Matrix: \n");
			display(A);
			fprintf("Observation Probability Matrix: \n");
			display(B);

			fprintf("Trained Model Parameters: \n");
			fprintf("Initial State Distribution: \n");
			display(HMM.pi);
			fprintf("State Transition Probability Matrix: \n");
			display(HMM.A);
			fprintf("Observation Probability Matrix: \n");
			display(HMM.B);

			String HMMModelFilePath = "HMMModel.dat";
			HMM.saveModel(HMMModelFilePath);
		}
		/*HMM.setPi(pi);
		HMM.setA(A);
		HMM.setB(B);*/	

		// Predict the single best state path
		
		// int[] O = new int[] {1, 0, 1, 1, 1, 0, 0, 1};
		
		int ID = new Random().nextInt(D);
		int[] O = Os[ID];
				
		HMM HMMt = new HMM();
		HMMt.loadModel("HMMModel.dat");
		int[] Q = HMMt.predict(O);

		fprintf("Observation sequence: \n");
		HMMt.showObservationSequence(O);
		fprintf("True state sequence: \n");
		HMMt.showStateSequence(Qs[ID]);
		fprintf("Predicted state sequence: \n");
		HMMt.showStateSequence(Q);
		double p = HMMt.evaluate(O);
		System.out.format("P(O|Theta) = %f\n", p);

	}

	/**
	 * Number of data sequences for training.
	 */
	int D;
	
	/**
	 * An integer array holding the length of each training sequence.
	 */
	int[] lengths;
	
	/**
	 * Observation sequences for training.
	 * Os[n][t] = O_t^n, n = 0,...,D - 1, t = 0,...,T_n - 1.
	 */
	int[][] Os;

	/**
	 * Hidden state sequences for training data.
	 * Qs[n][t] = q_t^n, n = 0,...,D - 1, t = 0,...,T_n - 1.
	 */
	int[][] Qs;

	// **************** Model Parameters: **************** //

	/**
	 * Number of states in the model.
	 */
	int N;

	/**
	 * Number of distinct observation symbols per state.
	 */
	int M;

	/**
	 * Initial state distribution. pi[i] = P(q_1 = S_i).
	 */
	double[] pi;

	/**
	 * State transition probability matrix.
	 * A[i][j] = P(q_{t+1} = S_j|q_t = S_i).
	 */
	double[][] A;

	/**
	 * Observation probability matrix. B[j][k] = P(v_k|S_j).
	 */
	double[][] B;

	// *************************************************** //

	/**
	 * Convergence precision.
	 */
	double epsilon;

	/**
	 * Maximal number of iterations.
	 */
	int maxIter;

	/**
	 * Default constructor.
	 */
	public HMM() {
		N = 0;
		M = 0;
		pi = null;
		A = null;
		B = null;
		Os = null;
		Qs = null;
		epsilon = 1e-3;
		maxIter = 500;
	}

	/**
	 * Construct an HMM.
	 * 
	 * @param N number of states in the model
	 * 
	 * @param M number of distinct observation symbols per state
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param maxIter maximal number of iterations
	 * 
	 */
	public HMM(int N, int M, double epsilon, int maxIter) {
		this.N = N;
		this.M = M;
		pi = new double[N];
		for (int i = 0; i < N; i++) {
			pi[i] = 0;
		}
		A = new double[N][];
		for (int i = 0; i < N; i++) {
			A[i] = new double[N];
			for (int j = 0; j < N; j++)
				A[i][j] = 0;
		}
		B = new double[N][];
		for (int i = 0; i < N; i++) {
			B[i] = new double[M];
			for (int k = 0; k < M; k++)
				B[i][k] = 0;
		}
		Os = null;
		Qs = null;

		this.epsilon = epsilon;
		this.maxIter = maxIter;
	}

	/**
	 * Construct an HMM with default convergence precision being
	 * 1e-6 and maximal number of iterations being 1000.
	 * 
	 * @param N number of states in the model
	 * 
	 * @param M number of distinct observation symbols per state
	 * 
	 */
	public HMM(int N, int M) {
		this.N = N;
		this.M = M;
		pi = new double[N];
		for (int i = 0; i < N; i++) {
			pi[i] = 0;
		}
		A = new double[N][];
		for (int i = 0; i < N; i++) {
			A[i] = new double[N];
			for (int j = 0; j < N; j++)
				A[i][j] = 0;
		}
		B = new double[N][];
		for (int i = 0; i < N; i++) {
			B[i] = new double[M];
			for (int k = 0; k < M; k++)
				B[i][k] = 0;
		}
		Os = null;
		Qs = null;

		epsilon = 1e-6;
		maxIter = 1000;
	}

	/**
	 * Feed observation sequences for training.
	 * Os[n][t] = O_t^n, n = 0,...,D - 1, t = 0,...,T_n - 1.
	 * 
	 * @param Os observation sequences
	 *  
	 */
	public void feedData(int[][] Os) {
		this.Os = Os;
	}

	/**
	 * Feed state sequences for training data.
	 * Qs[n][t] = q_t^n, n = 0,...,D - 1, t = 0,...,T_n - 1.
	 * 
	 * @param Qs state sequences
	 * 
	 */
	public void feedLabels(int[][] Qs) {
		this.Qs = Qs;
	}

	/**
	 * Compute P(O|Theta), the probability of the observation
	 * sequence given the model, by forward recursion without
	 * scaling.
	 * 
	 * @param O an observation sequence
	 * 
	 * @return P(O|Theta)
	 * 
	 */
	public double evaluate2(int[] O) {

		// Forward Recursion

		int T = O.length;
		double[] alpha_t = new double[N];
		for (int i = 0; i < N; i++) {
			alpha_t[i] = pi[i] * B[i][O[0]];
		}
		double[] alpha_t_plus_1 = new double[N];
		double[] temp = null;

		double sum = 0;
		int t = 1;
		do {
			for (int j = 0; j < N; j++) {
				sum = 0;
				for (int i = 0; i < N; i++) {
					sum = sum + alpha_t[i] * A[i][j] * B[j][O[t]];
				}
				alpha_t_plus_1[j] = sum;
			}
			temp = alpha_t;
			alpha_t = alpha_t_plus_1;
			alpha_t_plus_1 = temp;
			t = t + 1;
		} while (t < T);

		return sum(alpha_t);

	}

	/**
	 * Compute P(O|Theta), the probability of the observation
	 * sequence given the model, by forward recursion with
	 * scaling.
	 * 
	 * @param O an observation sequence
	 * 
	 * @return P(O|Theta)
	 * 
	 */
	public double evaluate(int[] O) {

		// Forward Recursion with Scaling

		int T = O.length;
		double[] c = allocateVector(T);
		double[] alpha_hat_t = allocateVector(N);
		double[] alpha_hat_t_plus_1 = allocateVector(N);
		double[] temp_alpha = null;
		double log_likelihood = 0;

		for (int t = 0; t < T; t++) {
			if (t == 0) {
				for (int i = 0; i < N; i++) {
					alpha_hat_t[i] = pi[i] * B[i][O[0]];
				}
			} else {
				clearVector(alpha_hat_t_plus_1);
				for (int j = 0; j < N; j++) {
					for (int i = 0; i < N; i++) {
						alpha_hat_t_plus_1[j] += alpha_hat_t[i] * A[i][j] * B[j][O[t]];
					}
				}
				temp_alpha = alpha_hat_t;
				alpha_hat_t = alpha_hat_t_plus_1;
				alpha_hat_t_plus_1 = temp_alpha;
			}
			c[t] = 1.0 / sum(alpha_hat_t);
			timesAssign(alpha_hat_t, c[t]);
			log_likelihood -= Math.log(c[t]);
		}

		return Math.exp(log_likelihood);
	}

	/**
	 * Compute the maximum argument.
	 * 
	 * @param V a {@code double} array
	 * 
	 * @return maximum argument
	 * 
	 *//*
	public int argmax(double[] V) {

		int maxIdx = 0;
		double maxVal = V[0];
		for (int i = 1; i < V.length; i++) {
			if (maxVal < V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;

	}*/

	/**
	 * Predict the best single state path for a given observation sequence
	 * using Viterbi algorithm.
	 * 
	 * @param O an observation sequence
	 * 
	 * @return the most probable state path
	 * 
	 */
	public int[] predict2(int[] O) {

		int T = O.length;
		int[] Q = new int[T];

		double[] delta_t = new double[N];
		double[] delta_t_plus_1 = new double[N];

		int[][] psi = new int[T][];
		for (int t = 0; t < T; t++) {
			psi[t] = new int[N]; 
		}

		double[] V = new double[N];

		// Viterbi algorithm

		for (int i = 0; i < N; i++) {
			delta_t[i] = pi[i] * B[i][O[1 - 1]];
		}

		int t = 1;
		int maxIdx = -1;
		double maxVal = 0;
		do {
			for (int j = 0; j < N; j++) {
				for (int i = 0; i < N; i++) {
					V[i] = delta_t[i] * A[i][j];
				}
				maxIdx = argmax(V);
				maxVal = V[maxIdx];
				delta_t_plus_1[j] = maxVal * B[j][O[t + 1 - 1]];
				psi[t + 1 - 1][j] = maxIdx;
			}
			// swap(delta_t, delta_t_plus_1);
			double[] temp = null;
			temp = delta_t;
			delta_t = delta_t_plus_1;
			delta_t_plus_1 = temp;
			t = t + 1;
		} while (t < T);

		// display(psi);

		int i_t = argmax(delta_t);
		Q[T - 1] = i_t;
		t = T;
		do {
			i_t = psi[t - 1][i_t];
			Q[t - 1 - 1] = i_t;
			t = t - 1;
		} while (t > 1);

		return Q;

	}

	/**
	 * Predict the best single state path for a given observation sequence
	 * using Viterbi algorithm with logarithms.
	 * 
	 * @param O an observation sequence
	 * 
	 * @return the most probable state path
	 * 
	 */
	public int[] predict(int[] O) {

		int T = O.length;
		int[] Q = new int[T];

		double[] phi_t = allocateVector(N);
		double[] phi_t_plus_1 = allocateVector(N);
		double[] temp_phi = null;

		int[][] psi = new int[T][];
		for (int t = 0; t < T; t++) {
			psi[t] = new int[N];
		}

		double[] V = allocateVector(N);

		// Viterbi algorithm using logarithms

		for (int i = 0; i < N; i++) {
			phi_t[i] = log(pi[i]) + log(B[i][O[0]]);
		}

		int t = 1;
		int maxIdx = -1;
		double maxVal = 0;
		do {
			for (int j = 0; j < N; j++) {
				for (int i = 0; i < N; i++) {
					V[i] = phi_t[i] + log(A[i][j]);
				}
				maxIdx = argmax(V);
				maxVal = V[maxIdx];
				phi_t_plus_1[j] = maxVal + log(B[j][O[t]]);
				psi[t][j] = maxIdx;
			}
			temp_phi = phi_t;
			phi_t = phi_t_plus_1;
			phi_t_plus_1 = temp_phi;
			t = t + 1;
		} while (t < T);

		// display(psi);

		int i_t = argmax(phi_t);
		Q[T - 1] = i_t;
		t = T;
		do {
			i_t = psi[t - 1][i_t];
			Q[t - 1 - 1] = i_t;
			t = t - 1;
		} while (t > 1);

		return Q;

	}

	/**
	 * Inference the basic HMM with scaling. Memory complexity
	 * is O(TN) + O(N^2) + O(NM), and computation complexity is
	 * O(tDTN^2), where t is the number of outer iterations.
	 */
	public void train() {

		int D = Os.length;
		int T_n = 0;
		double log_likelihood = 0;
		double log_likelihood_new = 0;
		double epsilon = this.epsilon;
		int maxIter = this.maxIter;

		// Initialization

		clearVector(pi);
		clearMatrix(A);
		clearMatrix(B);

		double[] a = allocateVector(N);
		double[] b = allocateVector(N);

		int[] Q_n = null;
		int[] O_n = null;
		
		if (Qs == null) {
			
			pi = initializePi();
			A = initializeA();
			B = initializeB();
			
		} else {
			
			for (int n = 0; n < D; n++) {
				Q_n = Qs[n];
				O_n = Os[n];
				T_n = Os[n].length;
				for (int t = 0; t < T_n; t++) {
					if (t < T_n - 1) {
						A[Q_n[t]][Q_n[t + 1]] += 1;
						a[Q_n[t]] += 1;
						if (t == 0) {
							pi[Q_n[0]] += 1;
						}
					}
					B[Q_n[t]][O_n[t]] += 1;
					b[Q_n[t]] += 1;
				}
			}
			divideAssign(pi, D);
			for (int i = 0; i < N; i++) {
				divideAssign(A[i], a[i]);
				divideAssign(B[i], b[i]);
			}
			
		}

		int s = 0;
		double[] pi_new = allocateVector(N);
		double[][] A_new = allocateMatrix(N, N);
		double[][] B_new = allocateMatrix(N, M);
		double[] temp_pi = null;
		double[][] temp_A = null;
		double[][] temp_B = null;
		double[][] alpha_hat = null;
		double[][] beta_hat = null;
		double[] c_n = null;
		double[][] xi = allocateMatrix(N, N);
		double[] gamma = allocateVector(N);
		do {

			// Clearance
			clearVector(pi_new);
			clearMatrix(A_new);
			clearMatrix(B_new);
			clearVector(a);
			clearVector(b);
			/*clearMatrix(xi);
			clearVector(gamma);*/
			log_likelihood_new = 0;

			for (int n = 0; n < D; n++) {

				// Q_n = Qs[n];
				O_n = Os[n];
				T_n = Os[n].length;
				c_n = allocateVector(T_n);
				alpha_hat = allocateMatrix(T_n, N);
				beta_hat = allocateMatrix(T_n, N);

				// Forward Recursion with Scaling			

				for (int t = 0; t <= T_n - 1; t++) {
					if (t == 0) {
						for (int i = 0; i < N; i++) {
							alpha_hat[0][i] = pi[i] * B[i][O_n[0]];
						}
					} else {
						for (int j = 0; j < N; j++) {
							for (int i = 0; i < N; i++) {
								alpha_hat[t][j] += alpha_hat[t - 1][i] * A[i][j] * B[j][O_n[t]];
							}
						}
					}
					c_n[t] = 1.0 / sum(alpha_hat[t]);
					timesAssign(alpha_hat[t], c_n[t]);
				}

				// Backward Recursion with Scaling

				for (int t = T_n + 1; t >= 2; t--) {
					if (t == T_n + 1) {
						for (int i = 0; i < N; i++) {
							beta_hat[t - 2][i] = 1;
						}
					}
					if (t <= T_n) {
						for (int i = 0; i < N; i++) {
							for (int j = 0; j < N; j++) {
								beta_hat[t - 2][i] += A[i][j] * B[j][O_n[t - 1]] * beta_hat[t - 1][j];
							}
						}
					}
					timesAssign(beta_hat[t - 2], c_n[t - 2]);
				}

				// Expectation Variables and Updating Model Parameters

				for (int t = 0; t <= T_n - 1; t++) {
					if (t < T_n - 1) {
						for (int i = 0; i < N; i++) {
							for (int j = 0; j < N; j++) {
								xi[i][j] = alpha_hat[t][i] * A[i][j] * B[j][O_n[t + 1]] * beta_hat[t + 1][j];
								// A_new[i][j] += xi[i][j];
							}
							plusAssign(A_new[i], xi[i]);
							gamma[i] = sum(xi[i]);
						}
						if (t == 0) {
							plusAssign(pi_new, gamma);
						}
						plusAssign(a, gamma);
					} else {
						assignVector(gamma, alpha_hat[t]);
					}
					for (int j = 0; j < N; j++) {
						B_new[j][O_n[t]] += gamma[j];
					}
					plusAssign(b, gamma);
					log_likelihood_new += -Math.log(c_n[t]);
				}

			}

			// Normalization (Sum to One)

			sum2one(pi_new);

			for (int i = 0; i < N; i++) {
				divideAssign(A_new[i], a[i]);
			}

			for (int j = 0; j < N; j++) {
				divideAssign(B_new[j], b[j]);
			}

			temp_pi = pi;
			pi = pi_new;
			pi_new = temp_pi;

			temp_A = A;
			A = A_new;
			A_new = temp_A;

			temp_B = B;
			B = B_new;
			B_new = temp_B;
			// display(B);

			s = s + 1;

			if (s > 1) {
				if (Math.abs((log_likelihood_new - log_likelihood) / log_likelihood) < epsilon) {
					fprintf("log[P(O|Theta)] does not increase.\n\n");
					break;
				}
			}

			log_likelihood = log_likelihood_new;
			fprintf("Iter: %d, log[P(O|Theta)]: %f\n", s, log_likelihood);

		} while (s < maxIter);

	}

	/**
	 * Generate a discrete distribution with sample size of n.
	 * 
	 * @param n sample size
	 * 
	 * @return a double array with sum 1
	 */
	public double[] genDiscreteDistribution(int n) {
		Random generator = null;
		generator = new Random();
		double[] res = allocateVector(n);
		do {
			for (int i = 0; i < n; i++) {
				res[i] = generator.nextDouble();
			}
		} while (sum(res) == 0);
		divideAssign(res, sum(res));
		return res;
	}

	private double[][] initializeB() {
		double[][] res = new double[N][];
		for (int i = 0; i < N; i++) {
			res[i] = genDiscreteDistribution(M);;
		}
		return res;
	}

	private double[][] initializeA() {
		double[][] res = new double[N][];
		for (int i = 0; i < N; i++) {
			res[i] = genDiscreteDistribution(N);
		}
		return res;
	}

	private double[] initializePi() {
		return genDiscreteDistribution(N);
	}

	/**
	 * Assign a 1D {@code double} array by a real scalar.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 *//*
	public void assignVector(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = v;
	}

	*//**
	 * Clear all elements of a 1D {@code double} array to zero.
	 * 
	 * @param V a {@code double} array
	 * 
	 *//*
	public void clearVector(double[] V) {
		assignVector(V, 0);
	}

	*//**
	 * Clear all elements of a 2D {@code double} array to zero.
	 * 
	 * @param M a 2D {@code double} array
	 * 
	 *//*
	public void clearMatrix(double[][] M) {
		for (int i = 0; i < M.length; i++) {
			assignVector(M[i], 0);
		}
	}

	*//**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 *//*
	public double[] allocateVector(int n) {
		double[] res = new double[n];
		assignVector(res, 0);
		return res;
	}

	*//**
	 * Allocate memory for a 2D {@code double} array.
	 * 
	 * @param nRows number of rows
	 * 
	 * @param nCols number of columns
	 * 
	 * @return a nRows by nCols 2D {@code double} array
	 * 
	 *//*
	public double[][] allocateMatrix(int nRows, int nCols) {
		double[][] res = new double[nRows][];
		for (int i = 0; i < nRows; i++) {
			res[i] = allocateVector(nCols);
		}
		return res;
	}

	*//**
	 * Element-wise division and assignment operation. It divides
	 * the first argument with the second argument and assign
	 * the result to the first argument, i.e., V = V / v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 *//*
	public void divideAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] / v;
	}

	*//**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V = V * v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 *//*
	public void timesAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] * v;
	}

	*//**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 .* V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 *//*
	public void timesAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] * V2[i];
	}

	*//**
	 * Compute the sum of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return sum(V)
	 * 
	 *//*
	public double sum(double[] V) {
		double res = 0;
		for (int i = 0; i < V.length; i++)
			res += V[i];
		return res;
	}

	*//**
	 * Sum a 1D {@code double} array to one, i.e., V[i] = V[i] / sum(V).
	 * 
	 * @param V a 1D {@code double} array
	 *//*
	public void sum2one(double[] V) {
		divideAssign(V, sum(V));
	}

	*//**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument with the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 + V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 *//*
	public void plusAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] + V2[i];
	}

	*//**
	 * Element-wise assignment operation. It assigns the first argument
	 * with the second argument, i.e., V1 = V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 *//*
	public void assignVector(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V2[i];
	}*/

	/**
	 * Save this HMM model to a file.
	 * 
	 * @param filePath file path to save the model
	 * 
	 */
	public void saveModel(String filePath) {

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(new HMMModel(pi, A, B));
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Load an HMM model from a file.
	 * 
	 * @param filePath file Path to load an HMM model from
	 * 
	 */
	public void loadModel(String filePath) {

		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			HMMModel HMMModel = (HMMModel)ois.readObject();
			N = HMMModel.N;
			M = HMMModel.M;
			pi = HMMModel.pi;
			A = HMMModel.A;
			B = HMMModel.B;
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

	public void setQs(int[][] Qs) {
		this.Qs = Qs;
	}

	public void setOs(int[][] Os) {
		this.Os = Os;
	}

	public void setPi(double[] pi) {
		this.pi = pi;
	}

	public void setA(double[][] A) {
		this.A = A;
	}

	public void setB(double[][] B) {
		this.B = B;
	}
	
	public double[] getPi() {
		return this.pi;
	}

	public double[][] getA() {
		return this.A;
	}

	public double[][] getB() {
		return this.B;
	}

	/**
	 * Show a state sequence.
	 * 
	 * @param Q a state sequence represented by a 1D
	 *          {@code int} array
	 * 
	 */
	public void showStateSequence(int[] Q) {
		for (int t = 0; t < Q.length; t++) {
			System.out.format("%d ", Q[t]);
		}
		System.out.println();
	}

	/**
	 * Show an observation sequence.
	 * 
	 * @param O an observation sequence represented by a 1D
	 *          {@code int} array
	 * 
	 */
	public void showObservationSequence(int[] O) {
		for (int t = 0; t < O.length; t++) {
			System.out.format("%d ", O[t]);
		}
		System.out.println();
	}

	/**
	 * Generate observation sequences with hidden state sequences
	 * given model parameters and number of data sequences.
	 * 
	 * @param D number of data sequences to be generated
	 * 
	 * @param T_min minimal sequence length
	 * 
	 * @param T_max maximal sequence length
	 * 
	 * @param pi initial state distribution
	 * 
	 * @param A state transition probability matrix
	 * 
	 * @param B observation probability matrix
	 * 
	 * @return a 3D integer array composed of two 2D integer array with
	 *         the first one being the observation sequences and second
	 *         one being the hidden state sequences
	 * 
	 */
	public static int[][][] generateDataSequences(int D, int T_min, int T_max, double[] pi, double[][] A, double[][] B) {

		int[][][] res = new int[2][][];

		int[][] Os = new int[D][];
		int[][] Qs = new int[D][];

		int N = A.length;
		int M = B[0].length;
		double[] distribution = null;
		double sum = 0;

		Random generator = new Random();
		double rndRealScalor = 0;

		for (int n = 0; n < D; n++) {

			int T_n = generator.nextInt(T_max - T_min + 1) + T_min;
			int[] O_n = new int[T_n];
			int[] Q_n = new int[T_n];

			for (int t = 0; t < T_n; t++) {

				rndRealScalor = generator.nextDouble();

				if (t == 0) { // Initial state
					distribution = pi;
				} else { // Following states
					distribution = A[Q_n[t - 1]];
				}

				// Generate a state sequence
				sum = 0;
				for (int i = 0; i < N; i++) {
					sum += distribution[i];
					if (rndRealScalor <= sum) {
						Q_n[t] = i;
						break;
					}
				}

				rndRealScalor = generator.nextDouble();

				// Generate an observation sequence
				distribution = B[Q_n[t]];
				sum = 0;
				for (int k = 0; k < M; k++) {
					sum += distribution[k];
					if (rndRealScalor <= sum) {
						O_n[t] = k;
						break;
					}
				}

			}
			Os[n] = O_n;
			Qs[n] = Q_n;
		}

		res[0] = Os;
		res[1] = Qs;

		return res;

	}

}

/***
 * HMM model parameters.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 15th, 2013
 */
class HMMModel implements Serializable{

	// **************** Model Parameters: **************** //

	/**
	 * 
	 */
	private static final long serialVersionUID = -3585978995931113277L;

	/**
	 * Number of states in the model.
	 */
	public int N;

	/**
	 * Number of distinct observation symbols per state.
	 */
	public int M;

	/**
	 * Initial state distribution. pi[i] = P(q_1 = S_i).
	 */
	public double[] pi;

	/**
	 * State transition probability matrix.
	 * A[i][j] = P(q_{t+1} = S_j|q_t = S_i).
	 */
	public double[][] A;

	/**
	 * Observation probability matrix. B[j][k] = P(v_k|S_j).
	 */
	public double[][] B;

	// *************************************************** //

	/**
	 * Constructor for an HMM model.
	 * 
	 * @param pi initial state distribution
	 * 
	 * @param A state transition probability matrix
	 * 
	 * @param B observation probability matrix
	 * 
	 */
	public HMMModel(double[] pi, double[][] A, double[][] B) {
		this.pi = pi;
		this.A = A;
		this.B = B;
		this.N = A.length;
		this.M = B[0].length;
	}

}
