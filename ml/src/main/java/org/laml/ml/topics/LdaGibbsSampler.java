/*
 * (C) Copyright 2005, Gregor Heinrich (gregor :: arbylon : net) (This file is
 * part of the org.knowceans experimental software packages.)
 */
/*
 * LdaGibbsSampler is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option) any
 * later version.
 */
/*
 * LdaGibbsSampler is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 */
/*
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 */

/*
 * Created on Mar 6, 2005
 */
package org.laml.ml.topics;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.TreeMap;

import org.laml.la.matrix.Matrix;
import org.laml.ml.options.LDAOptions;

/**
 * Gibbs sampler for estimating the best assignments of topics for words and
 * documents in a corpus. The algorithm is introduced in Tom Griffiths' paper
 * "Gibbs sampling in the generative model of Latent Dirichlet Allocation"
 * (2002).
 * 
 * @author heinrich
 */
public class LdaGibbsSampler {
	
	/**
     * Driver with example data.
     * 
     * @param args
     */
    public static void main(String[] args) {

        // words in documents
        int[][] documents = { {1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6},
            {2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2},
            {1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0},
            {5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0},
            {2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0},
            {5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2}};
        // vocabulary
        int V = 7;
        //int M = documents.length;
        
        // # topics
        int K = 2;
        // good values alpha = 2, beta = .5
        double alpha = 2;
        double beta = .5;

        System.out.println("Latent Dirichlet Allocation using Gibbs Sampling.");

        LdaGibbsSampler LDA = new LdaGibbsSampler(documents, V);
        LDA.configure(10000, 2000, 100, 10);
        LDA.gibbs(K, alpha, beta);

        double[][] theta = LDA.getTheta();
        double[][] phi = LDA.getPhi();

        System.out.println();
        System.out.println();
        System.out.println("Document--Topic Associations, Theta[d][k] (alpha="
            + alpha + ")");
        System.out.print("d\\k\t");
        for (int m = 0; m < theta[0].length; m++) {
            System.out.print("   " + m % 10 + "    ");
        }
        System.out.println();
        for (int m = 0; m < theta.length; m++) {
            System.out.print(m + "\t");
            for (int k = 0; k < theta[m].length; k++) {
                // System.out.print(theta[m][k] + " ");
                System.out.print(shadeDouble(theta[m][k], 1) + " ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Topic--Term Associations, Phi[k][w] (beta=" + beta
            + ")");

        System.out.print("k\\w\t");
        for (int w = 0; w < phi[0].length; w++) {
            System.out.print("   " + w % 10 + "    ");
        }
        System.out.println();
        for (int k = 0; k < phi.length; k++) {
            System.out.print(k + "\t");
            for (int w = 0; w < phi[k].length; w++) {
                // System.out.print(phi[k][w] + " ");
                System.out.print(shadeDouble(phi[k][w], 1) + " ");
            }
            System.out.println();
        }
    }

    Corpus corpus;
	
	LDAOptions LDAOptions;

    /**
     * document data (word lists)
	 * documents[m][n] is the term index in the vocabulary for the n-th word of the m-th document
     */
    int[][] documents;

    /**
     * vocabulary size
     */
    int V;

    /**
     * number of topics
     */
    int K;

    /**
     * Dirichlet parameter (document--topic associations)
     */
    double alpha;

    /**
     * Dirichlet parameter (topic--term associations)
     */
    double beta;

    /**
     * topic assignments for each word.
     */
    int z[][];

    /**
     * nw[i][j] number of instances of term i assigned to topic j.
     */
    int[][] nw;

    /**
     * nd[i][j] number of words in document i assigned to topic j.
     */
    int[][] nd;

    /**
     * nwsum[j] total number of words assigned to topic j.
     */
    int[] nwsum;

    /**
     * ndsum[i] total number of words in document i.
     */
    int[] ndsum;

    /**
     * cumulative statistics of theta
     */
    double[][] thetasum;

    /**
     * cumulative statistics of phi
     */
    double[][] phisum;

    /**
     * size of statistics
     */
    int numstats;

    /**
     * sampling lag (?)
     */
    private static int THIN_INTERVAL = 20;

    /**
     * burn-in period
     */
    private static int BURN_IN = 100;

    /**
     * max iterations
     */
    private static int ITERATIONS = 1000;

    /**
     * sample lag (if -1 only one sample taken)
     */
    private static int SAMPLE_LAG;

    private static int dispcol = 0;

    /**
     * Initialize the Gibbs sampler with data.
     * 
     * @param documents a 2D integer array where documents[m][n] is
	 *                  the term index in the vocabulary for the n-th
	 *                  word of the m-th document. Indices always start
	 *                  from 0.
     * 
     * @param V
     *            vocabulary size
     */
    public LdaGibbsSampler(int[][] documents, int V) {

        this.documents = documents;
        this.V = V;
    }
    
    public LdaGibbsSampler(LDAOptions LDAOptions) {
    	this.documents = null;
    	this.V = 0;
    	this.corpus = new Corpus();
    	this.LDAOptions = LDAOptions;
    }
    
    public LdaGibbsSampler() {
    	this.documents = null;
    	this.V = 0;
    	this.corpus = new Corpus();
    }

    /**
	 * Load {@code corpus} and {@code documents} from a {@code ArrayList<TreeMap<Integer, Integer>>} instance.
	 * Each element of the {@code ArrayList} is a doc-term count mapping.
	 * 
	 * @param docTermCountArray
	 *        A {@code ArrayList<TreeMap<Integer, Integer>>} instance,
	 *        each element of the {@code ArrayList} records the doc-term
	 *        count mapping for the corresponding document.
	 */
	public void readCorpusFromDocTermCountArray(ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
		
		corpus.readCorpusFromDocTermCountArray(docTermCountArray);
		
		this.documents = corpus.documents;
		this.V = corpus.nTerm;
		
	}
	
	/**
	 * Load {@code corpus} and {@code documents} from a LDAInput file. 
	 * 
	 * @param LDAInputDataFilePath
	 *        The file path specifying the path of the LDAInput file.
	 */
	public void readCorpusFromLDAInputFile(String LDAInputDataFilePath) {
		corpus.readCorpusFromLDAInputFile(LDAInputDataFilePath);
		this.documents = corpus.documents;
		this.V = corpus.nTerm;
	}
	
	/**
     * Load {@code corpus} and {@code documents} from a text file located at {@code String} docTermCountFilePath.
     * 
     * @param docTermCountFilePath
     *        A {@code String} specifying the location of the text file holding doc-term-count matrix data.
     */
	public void readCorpusFromDocTermCountFile(String docTermCountFilePath) {
		corpus.readCorpusFromDocTermCountFile(docTermCountFilePath);
		documents = corpus.documents;
		V = corpus.nTerm;
	}
	
	/**
	 * Load {@code corpus} and {@code documents} from a {@code RealMatrix} instance.
	 * 
	 * @param X a matrix with each column being a term count vector for a document
	 *          with X(i, j) being the number of occurrence for the i-th vocabulary
	 *          term in the j-th document
	 *          
	 */
	public void readCorpusFromMatrix(Matrix X) {
		corpus.readCorpusFromMatrix(X);
		documents = corpus.documents;
		V = corpus.nTerm;
	}
	
    /**
     * Initialization: Must start with an assignment of observations to topics ?
     * Many alternatives are possible, I chose to perform random assignments
     * with equal probabilities. 
     * 
     * @param K
     *            number of topics
     * 
     * @return assignment of topics to words
     * 
     */
    public int[][] initialState(int K) {

        int M = documents.length;

        // Initialize count variables.
        nw = new int[V][K];
        nd = new int[M][K];
        nwsum = new int[K];
        ndsum = new int[M];

        // The z_i are are initialized to values in [1,K] to determine the
        // initial state of the Markov chain.

        z = new int[M][];
        for (int m = 0; m < M; m++) {
            int N = documents[m].length;
            z[m] = new int[N];
            for (int n = 0; n < N; n++) {
                int topic = (int) (Math.random() * K);
                z[m][n] = topic;
                // number of instances of term v assigned to topic j
                nw[documents[m][n]][topic]++;
                // number of words in document i assigned to topic j.
                nd[m][topic]++;
                // total number of words assigned to topic j.
                nwsum[topic]++;
            }
            // total number of words in document i
            ndsum[m] = N;
        }
        
        return z;
        
    }

    /**
     * Main method: Select initial state ? Repeat a large number of times: 1.
     * Select an element 2. Update conditional on other elements. If
     * appropriate, output summary for each run.
     * 
     * @param K
     *            number of topics
     * @param alpha
     *            symmetric prior parameter on document--topic associations
     * @param beta
     *            symmetric prior parameter on topic--term associations
     */
    public void gibbs(int K, double alpha, double beta) {
    	
        this.K = K;
        this.alpha = alpha;
        this.beta = beta;

        // init sampler statistics
        if (SAMPLE_LAG > 0) {
            thetasum = new double[documents.length][K];
            phisum = new double[K][V];
            numstats = 0;
        }

        // initial state of the Markov chain:
        initialState(K);

        System.out.println("Sampling " + ITERATIONS
            + " iterations with burn-in of " + BURN_IN + " (B/S="
            + THIN_INTERVAL + ").");

        dispcol = 0;
        
        for (int i = 0; i < ITERATIONS; i++) {

            // for all z_i
            for (int m = 0; m < z.length; m++) {
                for (int n = 0; n < z[m].length; n++) {

                    // (z_i = z[m][n])
                    // sample from p(z_i|z_-i, w)
                    int topic = sampleFullConditional(m, n);
                    z[m][n] = topic;
                }
            }

            if ((i < BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("B");
                dispcol++;
            }
            // display progress
            if ((i > BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("S");
                dispcol++;
            }
            // get statistics after burn-in
            if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
                updateParams();
                System.out.print("|");
                if (i % THIN_INTERVAL != 0)
                    dispcol++;
            }
            if (dispcol >= 100) {
                System.out.println();
                dispcol = 0;
            }
        }
        
        System.out.println();
        
    }

    /**
     * Sample a topic z_i from the full conditional distribution: p(z_i = j |
     * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
     * alpha)/(n_-i,.(d_i) + K * alpha)
     * 
     * @param m
     *            document
     * @param n
     *            word
     */
    private int sampleFullConditional(int m, int n) {

        // remove z_i from the count variables
        int topic = z[m][n];
        nw[documents[m][n]][topic]--;
        nd[m][topic]--;
        nwsum[topic]--;
        ndsum[m]--;

        // do multinomial sampling via cumulative method:
        double[] p = new double[K];
        for (int k = 0; k < K; k++) {
            p[k] = (nw[documents[m][n]][k] + beta) / (nwsum[k] + V * beta)
                * (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
        }
        // cumulate multinomial parameters
        for (int k = 1; k < p.length; k++) {
            p[k] += p[k - 1];
        }
        // scaled sample because of unnormalised p[]
        double u = Math.random() * p[K - 1];
        for (topic = 0; topic < p.length; topic++) {
            if (u < p[topic])
                break;
        }

        // add newly estimated z_i to count variables
        nw[documents[m][n]][topic]++;
        nd[m][topic]++;
        nwsum[topic]++;
        ndsum[m]++;

        return topic;
    }

    /**
     * Add to the statistics the values of theta and phi for the current state.
     */
    private void updateParams() {
        for (int m = 0; m < documents.length; m++) {
            for (int k = 0; k < K; k++) {
                thetasum[m][k] += (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
            }
        }
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < V; t++) {
                phisum[k][t] += (nw[t][k] + beta) / (nwsum[k] + V * beta);
            }
        }
        numstats++;
    }

    /**
     * Retrieve estimated document--topic associations. If sample lag > 0 then
     * the mean value of all sampled statistics for theta[][] is taken.
     * 
     * @return theta multinomial mixture of document topics (M x K)
     */
    public double[][] getTheta() {
        double[][] theta = new double[documents.length][K];

        if (SAMPLE_LAG > 0) {
            for (int m = 0; m < documents.length; m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = thetasum[m][k] / numstats;
                }
            }

        } else {
            for (int m = 0; m < documents.length; m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
                }
            }
        }

        return theta;
    }

    /**
     * Retrieve estimated topic--word associations. If sample lag > 0 then the
     * mean value of all sampled statistics for phi[][] is taken.
     * 
     * @return phi multinomial mixture of topic words (K x V)
     */
    public double[][] getPhi() {
        double[][] phi = new double[K][V];
        if (SAMPLE_LAG > 0) {
            for (int k = 0; k < K; k++) {
                for (int t = 0; t < V; t++) {
                    phi[k][t] = phisum[k][t] / numstats;
                }
            }
        } else {
            for (int k = 0; k < K; k++) {
                for (int t = 0; t < V; t++) {
                    phi[k][t] = (nw[t][k] + beta) / (nwsum[k] + V * beta);
                }
            }
        }
        return phi;
    }

    /**
     * Print table of multinomial data
     * 
     * @param data
     *            vector of evidence
     * @param fmax
     *            max frequency in display
     * @return the scaled histogram bin values
     */
    public static double[] hist(double[] data, int fmax) {

        double[] hist = new double[data.length];
        // scale maximum
        double hmax = 0;
        for (int i = 0; i < data.length; i++) {
            hmax = Math.max(data[i], hmax);
        }
        double shrink = fmax / hmax;
        for (int i = 0; i < data.length; i++) {
            hist[i] = shrink * data[i];
        }

        NumberFormat nf = new DecimalFormat("00");
        String scale = "";
        for (int i = 1; i < fmax / 10 + 1; i++) {
            scale += "    .    " + i % 10;
        }

        System.out.println("x" + nf.format(hmax / fmax) + "\t0" + scale);
        for (int i = 0; i < hist.length; i++) {
            System.out.print(i + "\t|");
            for (int j = 0; j < Math.round(hist[i]); j++) {
                if ((j + 1) % 10 == 0)
                    System.out.print("]");
                else
                    System.out.print("|");
            }
            System.out.println();
        }
        
        return hist;
        
    }

    /**
     * Configure the gibbs sampler
     * 
     * @param iterations
     *            number of total iterations
     * @param burnIn
     *            number of burn-in iterations
     * @param thinInterval
     *            update statistics interval
     * @param sampleLag
     *            sample interval (-1 for just one sample at the end)
     */
    public void configure(int iterations, int burnIn, int thinInterval,
        int sampleLag) {
        ITERATIONS = iterations;
        BURN_IN = burnIn;
        THIN_INTERVAL = thinInterval;
        SAMPLE_LAG = sampleLag;
    }
    
	public void configure(LDAOptions LDAOptions) {
		ITERATIONS = LDAOptions.iterations;
		BURN_IN = LDAOptions.burnIn;
		THIN_INTERVAL = LDAOptions.thinInterval;
		SAMPLE_LAG = LDAOptions.sampleLag;
	}
	
	public void run() {
		configure(LDAOptions);
		gibbs(LDAOptions.nTopic, LDAOptions.alpha, LDAOptions.beta);
	}
	
	public void run(LDAOptions LDAOptions) {
		configure(LDAOptions);
		gibbs(LDAOptions.nTopic, LDAOptions.alpha, LDAOptions.beta);
	}
    
    public static void run(Corpus corpus, LDAOptions LDAOptions) {
    	// Vocabulary size
        int V = corpus.nTerm;
        //int M = documents.length;
        
        // # topics
        // int K = 10;
        // good values alpha = 2, beta = .5
        /*double alpha = 2;
        double beta = .5;*/
        
        int[][] documents = corpus.getDocuments();

        System.out.println("Latent Dirichlet Allocation using Gibbs Sampling.");

        LdaGibbsSampler LDA = new LdaGibbsSampler(documents, V);
        LDA.configure(500, 100, 50, 10);
        LDA.gibbs(LDAOptions.nTopic, LDAOptions.alpha, LDAOptions.beta);

        double[][] theta = LDA.getTheta();
        double[][] phi = LDA.getPhi();

        System.out.println();
        System.out.println();
        System.out.println("Document--Topic Associations, Theta[d][k] (alpha="
            + LDAOptions.alpha + ")");
        System.out.print("d\\k\t");
        for (int m = 0; m < theta[0].length; m++) {
            System.out.print("   " + m % 10 + "    ");
        }
        System.out.println();
        for (int m = 0; m < theta.length; m++) {
            System.out.print(m + "\t");
            for (int k = 0; k < theta[m].length; k++) {
                // System.out.print(theta[m][k] + " ");
                System.out.print(shadeDouble(theta[m][k], 1) + " ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Topic--Term Associations, Phi[k][w] (beta=" + LDAOptions.beta
            + ")");

        System.out.print("k\\w\t");
        for (int w = 0; w < phi[0].length; w++) {
            System.out.print("   " + w % 10 + "    ");
        }
        System.out.println();
        for (int k = 0; k < phi.length; k++) {
            System.out.print(k + "\t");
            for (int w = 0; w < phi[k].length; w++) {
                // System.out.print(phi[k][w] + " ");
                System.out.print(shadeDouble(phi[k][w], 1) + " ");
            }
            System.out.println();
        }
    }

    static String[] shades = {"     ", ".    ", ":    ", ":.   ", "::   ",
        "::.  ", ":::  ", ":::. ", ":::: ", "::::.", ":::::"};

    static NumberFormat lnf = new DecimalFormat("00E0");

    /**
     * create a string representation whose gray value appears as an indicator
     * of magnitude, cf. Hinton diagrams in statistics.
     * 
     * @param d
     *            value
     * @param max
     *            maximum value
     * @return a string representation for a value
     */
    public static String shadeDouble(double d, double max) {
        int a = (int) Math.floor(d * 10 / max + 0.5);
        if (a > 10 || a < 0) {
            String x = lnf.format(d);
            a = 5 - x.length();
            for (int i = 0; i < a; i++) {
                x += " ";
            }
            return "<" + x + ">";
        }
        return "[" + shades[a] + "]";
    }
}
