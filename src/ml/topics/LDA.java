package ml.topics;

import java.util.ArrayList;
import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.options.LDAOptions;

import static ml.utils.Printer.*;

/**
 * An interface for the Gibbs sampler implementation of LDA
 * by Gregor Heinrich.
 * 
 * @author Mingjie Qian
 * @version 1.0 June 20th, 2014
 */
public class LDA extends TopicModel {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// words in documents
		int[][] documents = { 
				{1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6},
				{2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2},
				{1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0},
				{5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0},
				{2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0},
				{5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2} 
		};
        
		LDAOptions LDAOptions = new LDAOptions();
		LDAOptions.nTopic = 2;
		LDAOptions.iterations = 5000;
		LDAOptions.burnIn = 1500;
		LDAOptions.thinInterval = 200;
		LDAOptions.sampleLag = 10;
		/*LDAOptions.alpha = 50.0 / LDAOptions.nTopic;
		LDAOptions.beta = 200.0 / textProcessor.getNTerm();*/
		LDAOptions.alpha = 2;
		LDAOptions.beta = 0.5;
		
		LDA LDA = new LDA(LDAOptions);
		LDA.readCorpus(documents);
		// display(LDA.dataMatrix);
		
		long start = System.currentTimeMillis();
		LDA.train();
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n\n", elapsedTime);
		
		fprintf("Topic--term associations: \n");
		display(LDA.topicMatrix);
		
		fprintf("Document--topic associations: \n");
		display(LDA.indicatorMatrix);
		
		
		Matrix X = Corpus.documents2Matrix(documents);
		TopicModel lda = new LDA(LDAOptions);
		lda.readCorpus(X);
		// display(lda.dataMatrix);
		
		start = System.currentTimeMillis();
		lda.train();
		elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n\n", elapsedTime);
		
		fprintf("Topic--term associations: \n");
		display(lda.topicMatrix);
		
		fprintf("Document--topic associations: \n");
		display(lda.indicatorMatrix);
		
	}

	LdaGibbsSampler gibbsSampler;
	
	public LDA(LDAOptions LDAOptions) {
		super(LDAOptions.nTopic);
		gibbsSampler = new LdaGibbsSampler(LDAOptions);
	}
	
	public LDA() {
	}

	public LDA(int nTopic) {
		super(nTopic);
	}

	@Override
	public void train() {
		gibbsSampler.run();
		topicMatrix = new DenseMatrix(gibbsSampler.getPhi()).transpose();
		indicatorMatrix = new DenseMatrix(gibbsSampler.getTheta());
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
	public void readCorpus(ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
		gibbsSampler.readCorpusFromDocTermCountArray(docTermCountArray);
		dataMatrix = Corpus.documents2Matrix(gibbsSampler.documents);
	}
	
	/**
	 * Load {@code corpus} and {@code documents} from a LDAInput file.
	 * Term indices must start from 0.
	 * 
	 * @param LDAInputDataFilePath
	 *        The file path specifying the path of the LDAInput file.
	 */
	public void readCorpus(String LDAInputDataFilePath) {
		gibbsSampler.readCorpusFromLDAInputFile(LDAInputDataFilePath);
		dataMatrix = Corpus.documents2Matrix(gibbsSampler.documents);
	}
	
	/**
     * Load {@code corpus} and {@code documents} from a text file located at {@code String} docTermCountFilePath.
     * 
     * @param docTermCountFilePath
     *        A {@code String} specifying the location of the text file holding doc-term-count matrix data.
     */
	public void readCorpusFromDocTermCountFile(String docTermCountFilePath) {
		gibbsSampler.readCorpusFromDocTermCountFile(docTermCountFilePath);
		dataMatrix = Corpus.documents2Matrix(gibbsSampler.documents);
	}
	
	/**
	 * Feed documents from a 2D integer array.
	 * 
	 * @param documents a 2D integer array where documents[m][n] is
	 *                  the term index in the vocabulary for the n-th
	 *                  word of the m-th document. Indices always start
	 *                  from 0.
	 */
	public void readCorpus(int[][] documents) {
		if (documents == null || documents.length == 0) {
			System.err.println("Empty documents!");
			System.exit(1);
		}
		gibbsSampler.documents = documents;
		gibbsSampler.V = Corpus.getVocabularySize(documents);
		dataMatrix = Corpus.documents2Matrix(documents);
	}
	
	/**
	 * Load {@code corpus} and {@code documents} from a {@code RealMatrix} instance.
	 * 
	 * @param X a matrix with each column being a term count vector for a document
	 *          with X(i, j) being the number of occurrence for the i-th vocabulary
	 *          term in the j-th document
	 *          
	 */
	public void readCorpus(Matrix X) {
		dataMatrix = X;
		gibbsSampler.readCorpusFromMatrix(X);
	}

}
