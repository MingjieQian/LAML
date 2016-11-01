package org.laml.ml.topics;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.laml.la.matrix.*;

import org.laml.la.utils.Utility;

/**
 * A class to model corpus. Term indices always start from 0, and are
 * used to index elements in a 2D integer array. Term IDs always start
 * from 1, and are used in a {@code Vector} of termID sequences.
 * 
 * @author Mingjie Qian
 * @version Jan. 17th, 2013
 */
public class Corpus {
	
	/**
	 * The starting index for LDA_Blei input data. Default is 0.
	 */
	public static int IdxStart = 0;
	
	/**
	 * A {@code Vector} of termID sequences. Each element of the vector is a sequence
	 * of termID (starting from 1) of a document. Each termID represents a corresponding
	 * term in the vocabulary. For example, assume a term occurs in a document ten times,
	 * then we have ten same termID for this term in the sequence.
	 */
	private Vector<Vector<Integer>> corpus;
	
	/** 
	 * A {@code ArrayList} of {@code TreeMap} storing the doc-term-count matrix.
	 * The {@code TreeMap} mapping a termID to its observed counts.
	 */
	public ArrayList<TreeMap<Integer, Integer>> docTermCountArray = new ArrayList<TreeMap<Integer, Integer>>();
	
	/**
	 * 2D integer array carrying the doc-term count matrix. documents[i][j]
	 * is the number of occurrence for the j-th vocabulary term in the i-th
	 * document. Term indices start from 0 for {@code documents}.
	 */
	public int[][] documents;
	
	/**
	 * Vocabulary size.
	 */
	public int nTerm;
	
	/**
	 * Number of documents in the corpus.
	 */
	public int nDoc;
	
	/**
	 * The total number of words in the corpus.
	 */
	// private int nTotalWord;
	
	/**
	 * Constructor for the class {@code Corpus}.
	 */
	public Corpus() {
		corpus = new Vector<Vector<Integer>>();
		documents = null;
		nTerm = 0;		
		nDoc = 0;		
		// nTotalWord = 0;
	}
	
	/**
	 * Clear corpus for class {@code Corpus}. 
	 */
	public void clearCorpus() {
		for ( int i = 0; i < corpus.size(); i++ ) {
			corpus.get(i).clear();
		}
		corpus.clear();
		nTerm = 0;		
		nDoc = 0;
		// nTotalWord = 0;
	}
	
	/**
     * Clear {@code docTermCountArray}.
     */
	public void clearDocTermCountArray() {
		if ( docTermCountArray.size() == 0 )
			return;
		Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
		while (iter.hasNext()) {
			iter.next().clear();
		}
		docTermCountArray.clear();
	}
	
	/**
	 * Get the documents.
	 * 
	 * @return {@code documents}.
	 */
	public int[][] getDocuments() {
		return documents;
	}
	
	/**
	 * Load {@code corpus} and {@code documents} from a LDAInput file. 
	 * 
	 * @param LDAInputDataFilePath
	 *        The file path specifying the path of the LDAInput file.
	 */
	public void readCorpusFromLDAInputFile(String LDAInputDataFilePath) {
		
		clearCorpus();
		
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(LDAInputDataFilePath));
		} catch (FileNotFoundException e) {
			System.out.println("Cannot open file: " + LDAInputDataFilePath);
			e.printStackTrace();
		}
		String line = "";
		int termID = 0, count = 0;
		//String token = "";
		int docID = 0;
		int nUniqueTerms = 0;
		String delimiters = " :\t";
		
		try {
			
			while ((line = br.readLine()) != null) {
				
				docID++;
				nDoc++;
				
				Vector<Integer> doc = new Vector<Integer>();
				
			    StringTokenizer tokenizer = new StringTokenizer(line);
			    
			    nUniqueTerms = Integer.parseInt(tokenizer.nextToken(delimiters));
			    
			    System.out.println("DocID: " + docID + ", nUniqueTerms: " + nUniqueTerms);
			    
			    while (tokenizer.hasMoreTokens()) {
			    	/* Note that termID will always start from 1 whether the
			    	 * indices in LDAInputFile start from 0 or 1.
			    	 */
			    	termID = Integer.parseInt(tokenizer.nextToken(delimiters)) + (1 - IdxStart);
			    	count = Integer.parseInt(tokenizer.nextToken(delimiters));
			    	for ( int i = 0; i < count; i++ ) {
			    		doc.add(termID);
			    		// nTotalWord++;
			    	}
			    	if ( termID > nTerm )
			    		nTerm = termID;
			    }
			    corpus.add(doc);
			}
			
			br.close();
			
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		documents = corpus2Documents(corpus);
		
	}
	
	/**
     * Load {@code corpus} and {@code documents} from a text file located at {@code String} docTermCountFilePath.
     * 
     * @param docTermCountFilePath
     *        A {@code String} specifying the location of the text file holding doc-term-count matrix data.
     */
	public void readCorpusFromDocTermCountFile(String docTermCountFilePath) {
		
		clearDocTermCountArray();
		clearCorpus();
		
		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;
		TreeMap<Integer, Integer> docTermCountMap = null;
		Vector<Integer> doc = null;
		// int docID_old = 0;
		int docID = 0;
		int termID = 0;
		int count = 0;
		// int nz = 0;	
				
		pattern = Pattern.compile("[(]([\\d]+), ([\\d]+)[)]: ([\\d]+)");
		
        try {

            br = new BufferedReader(new FileReader(docTermCountFilePath));
            
        } catch (FileNotFoundException e) {
			
			System.out.println("Cannot open file: " + docTermCountFilePath);
			e.printStackTrace();
			
		}    
		try {
			
			while ((line = br.readLine()) != null) {

				matcher = pattern.matcher(line);

				if (!matcher.find()) {
					System.out
							.println("Data format for the docTermCountFile should be: (docID, termID): count");
					System.exit(0);
				}

				docID = Integer.parseInt(matcher.group(1));
				// if (docID != docID_old) {
				if (docID != nDoc) {
					if (nDoc > 0) {
						docTermCountArray.add(docTermCountMap);
						corpus.add(doc);
						if (nTerm < docTermCountMap.lastKey().intValue()) {
							nTerm = docTermCountMap.lastKey().intValue();
						}
						System.out.println("DocID: " + nDoc + ", nUniqueTerms: " + docTermCountMap.size());
						
						/*for (int i = docID_old + 1; i < docID; i++) {
							docTermCountArray.add(new TreeMap<Integer, Integer>(
									new Utility.keyAscendComparator<Integer>()));
							corpus.add(new Vector<Integer>());
							System.out.println("DocID: " + ++nDoc + ", Empty");
						}*/
						
					}
					
					/*
					 * Sometime a document may have empty content after being filtered
					 * by stop words. 
					 */
					for (int i = nDoc + 1; i < docID; i++) {
						docTermCountArray.add(new TreeMap<Integer, Integer>(
								new Utility.keyAscendComparator<Integer>()));
						corpus.add(new Vector<Integer>());
						System.out.println("DocID: " + ++nDoc + ", Empty");					
					}
					
					docTermCountMap = new TreeMap<Integer, Integer>(
							new Utility.keyAscendComparator<Integer>());
					doc = new Vector<Integer>();
					nDoc++;
					
				}
				termID = Integer.parseInt(matcher.group(2));
				count = Integer.parseInt(matcher.group(3));

				docTermCountMap.put(termID, count);
				for ( int i = 0; i < count; i++ ) {
		    		doc.add(termID);
		    		// nTotalWord++;
		    	}
				// docID_old = docID;
				// nz++;
			}

			if (docTermCountMap != null) {
				docTermCountArray.add(docTermCountMap);
				corpus.add(doc);
				if (nTerm < docTermCountMap.lastKey().intValue()) {
					nTerm = docTermCountMap.lastKey().intValue();
				}
				System.out.println("DocID: " + nDoc + ", nUniqueTerms: " + docTermCountMap.size());
			}
			
			br.close();

		} catch (NumberFormatException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
		
		documents = corpus2Documents(corpus);
		
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
		
		clearCorpus();
		
		int count = 0;
		
		Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
		TreeMap<Integer, Integer> docTermCountMap = null;
		Vector<Integer> doc = null;
		
		
		while (iter.hasNext()) {
			
			docTermCountMap = iter.next();
			doc = new Vector<Integer>();
			nDoc++;
			for(int termID : docTermCountMap.keySet()) {
				count = docTermCountMap.get(termID);
				for ( int i = 0; i < count; i++ ) {
		    		doc.add(termID);
		    		// nTotalWord++;		    		
		    	}
				/*if ( termID > nTerm )
		    		nTerm = termID;*/
			}
			if (nTerm < docTermCountMap.lastKey().intValue()) {
				nTerm = docTermCountMap.lastKey().intValue();
			}
			corpus.add(doc);
			
		}
		
		documents = corpus2Documents(corpus);
		
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
		
		clearCorpus();
		
		int count = 0;
		int termID = 0;
		Vector<Integer> doc = null;
		
		int nDoc = X.getColumnDimension();
		nTerm = X.getRowDimension();
		if (X instanceof DenseMatrix) 
			for (int d = 0; d < nDoc; d++) {
				doc = new Vector<Integer>();
				for (int t = 0; t < nTerm; t++) {
					count = (int)X.getEntry(t, d);
					if (count != 0) {
						termID = t + 1;
						for ( int i = 0; i < count; i++ ) {
							doc.add(termID);   		
						}
					}
				}
				corpus.add(doc);
			}
		else if (X instanceof SparseMatrix) {
			int[] ir = null;
			int[] jc = null;
			double[] pr = null;
			ir = ((SparseMatrix) X).getIr();
			jc = ((SparseMatrix) X).getJc();
			pr = ((SparseMatrix) X).getPr();
			for (int j = 0; j < nDoc; j++) {
				doc = new Vector<Integer>();
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					termID = ir[k] + 1;
					// A[r][j] = pr[k]
					count = (int) pr[k];
					for ( int i = 0; i < count; i++ ) {
						doc.add(termID);   		
					}
				}
				corpus.add(doc);
			}
		}

		documents = corpus2Documents(corpus);
		
	}
	
	/**
	 * Convert a {@code Vector} of termID sequences into a 2D doc-term
	 * count array. Term IDs always start from 1.
	 * 
	 * @param corpus a {@code Vector} of termID sequences
	 * 
	 * @return a 2D integer array carrying the doc-term count matrix
	 * 
	 */
	public static int[][] corpus2Documents(Vector<Vector<Integer>> corpus) {
		int[][] documents = new int[corpus.size()][];
		for ( int i = 0; i < corpus.size(); i++ ) {
			documents[i] = new int[corpus.get(i).size()];
			for (int w = 0; w < corpus.get(i).size(); w++ ) {
				// Make sure that term indices in {@code documents} start from 0
				documents[i][w] = corpus.get(i).get(w) - 1;
			}
		}
		return documents;
	}
	
	/**
	 * Convert a 2D doc-term count array into a matrix.
	 * 
	 * @param documents a 2D integer array carrying the doc-term
	 *                  count matrix
	 *                  
	 * @return a matrix with each column being a term count vector
	 *         for a document with X(i, j) being the number of
	 *         occurrence for the i-th vocabulary term in the j-th
	 *         document 
	 *         
	 */
	public static Matrix documents2Matrix(int[][] documents) {
		
		if (documents == null || documents.length == 0) {
			System.err.println("Empty documents!");
			System.exit(1);
		}
		
		int N = documents.length;
		int V = getVocabularySize(documents);
		Matrix res = new SparseMatrix(V, N);
		
		int[] document = null;
		int termIdx = -1;
		for (int docIdx = 0; docIdx < documents.length; docIdx++) {
			document = documents[docIdx];
			for (int i = 0; i < document.length; i++) {
				termIdx = document[i];
				res.setEntry(termIdx, docIdx, res.getEntry(termIdx, docIdx) + 1);
			}
		}
		
		return res;
		
	}
	
	/**
	 * Get the vocabulary size.
	 * 
	 * @param documents a 2D integer array where documents[m][n] is
	 *                  the term index in the vocabulary for the n-th
	 *                  word of the m-th document. Indices always start
	 *                  from 0.
	 *                  
	 * @return vocabulary size
	 * 
	 */
	public static int getVocabularySize(int[][] documents) {
		
		int maxTermIdx = 0;
		for (int i = 0; i < documents.length; i++) {
			for (int j = 0; j < documents[i].length; j++) {
				if (maxTermIdx < documents[i][j]) {
					maxTermIdx = documents[i][j];
				}
			}
		}
		return maxTermIdx + 1;
		
	}
	
	/**
	 * Set term staring index for LDA input file.
	 * @param IdxStart
	 */
	public static void setLDATermIndexStart(int IdxStart) {
		Corpus.IdxStart = IdxStart;
	}
	
}
