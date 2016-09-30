package org.laml.la.io;

import static org.laml.la.utils.Matlab.full;
import static org.laml.la.utils.Matlab.sparse;
import static org.laml.la.utils.Printer.disp;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Printer.printMatrix;
import static org.laml.la.utils.Printer.sprintf;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.vector.DenseVector;
import org.laml.la.vector.SparseVector;
import org.laml.la.vector.Vector;
import org.laml.la.utils.Pair;

public class IO {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		fprintf("%d", (int)Math.floor(-0.2));
		
		// fprintf("%d", Math.round(-0.99));
		
		String line = " 1:2.3 ";
		// System.out.println(line.trim());
		System.out.println(System.lineSeparator());
		String separator = System.lineSeparator();
		String separator2 = sprintf("%n");
		if (separator.equals(separator2)) {
			System.out.println("System.lineSeparator() == sprintf(\"%n\")");
		}
		StringTokenizer tokenizer = new StringTokenizer(line, " \t\n\r\f");
		try {
			System.out.println(tokenizer.nextToken());
		} catch (NoSuchElementException e) {
			System.out.println("The line is empty.");
		}
		if (tokenizer.hasMoreTokens()) {
			System.out.println("The line has more tokens.");
			System.out.println(tokenizer.nextToken());
		} else {
			System.out.println("The line is empty.");
		}

		/*
		 * 10	0	0	0	0
		 * 3	9	0	0	0
		 * 0	7	8	7	0
		 * 3	0	8	7	0
		 * 0	0	0	0	0
		 */
		int[] rIndices = new int[] {0, 1, 3, 1, 2, 2, 3, 2, 3};
		int[] cIndices = new int[] {0, 0, 0, 1, 1, 2, 2, 3, 3};
		double[] values = new double[] {10, 3, 3, 9, 7, 8, 8, 7, 7};
		int numRows = 5;
		int numColumns = 5;
		int nzmax = rIndices.length;

		Matrix S = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		fprintf("S:\n");
		printMatrix(S, 4);

		Matrix A = full(S);
		fprintf("A:\n");
		printMatrix(A, 4);

		Matrix S2 = sparse(A);
		fprintf("S2:\n");
		printMatrix(S2, 4);

		String filePath = null;

		filePath = "SparseMatrix.txt";
		saveMatrix(S, filePath);
		fprintf("Loaded S:\n");
		printMatrix(loadMatrix(filePath));
		
		DataSet.writeDataSet(S, null, "Dataset.txt");
		DataSet dataSet = null;
		try {
			dataSet = DataSet.readDataSetFromFile("Dataset.txt");
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		printMatrix(dataSet.X);

		filePath = "DenseMatrix.txt";
		saveMatrix(A, filePath);
		fprintf("Loaded A:\n");
		printMatrix(loadMatrix(filePath));

		/*filePath = "JMLSparseMatrix.txt";
		fprintf("Loaded JML Sparse Matrix:\n");
		printMatrix(loadMatrix(filePath));*/

		// printMatrix(full(loadMatrix(filePath)));

		String dataMatrixFilePath = "CNN - DocTermCount.txt";

		Matrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		filePath = "CNN-DocTermCountMatrix.txt";
		saveMatrix(X.transpose(), filePath);
		
		int dim = 4;
		Vector V = new SparseVector(dim);
		
		for (int i = 0; i < dim; i++) {
			fprintf("V(%d):	%.2f\n", i + 1, V.get(i));
		}
		
		V.set(3, 4.5);
		fprintf("V(%d):	%.2f%n", 3 + 1, V.get(3));
		V.set(1, 2.3);
		fprintf("V(%d):	%.2f%n", 1 + 1, V.get(1));
		V.set(1, 3.2);
		fprintf("V(%d):	%.2f%n", 1 + 1, V.get(1));
		V.set(3, 2.5);
		fprintf("V(%d):	%.2f%n", 3 + 1, V.get(3));
		
		fprintf("V:%n");
		disp(V);
		
		filePath = "SparseVector.txt";
		saveVector(filePath, V);
		fprintf("Loaded V:\n");
		disp(loadVector(filePath));
		
		filePath = "DenseVector.txt";
		saveVector(filePath, full(V));
		fprintf("Loaded V:\n");
		disp(loadVector(filePath));

	}
	
	/**
	 * Write a matrix into a text file.
	 * 
	 * @param filePath file path to write a matrix into
	 * 
	 * @param A a real matrix
	 * 
	 */
	public static void save(String filePath, Matrix A) {
		saveMatrix(filePath, A);
	}
	
	/**
	 * Write a matrix into a text file.
	 * 
	 * @param A a real matrix
	 * 
	 * @param filePath file path to write a matrix into
	 * 
	 */
	public static void save(Matrix A, String filePath) {
		saveMatrix(A, filePath);
	}

	/**
	 * Write a matrix into a text file.
	 * 
	 * @param A a real matrix
	 * 
	 * @param filePath file path to write a matrix into
	 * 
	 */
	public static void saveMatrix(Matrix A, String filePath) {
		if (A instanceof DenseMatrix) {
			saveDenseMatrix(A, filePath);
		} else {
			saveSparseMatrix(A, filePath);
		}
	}

	/**
	 * Write a matrix into a text file.
	 * 
	 * @param filePath file path to write a matrix into
	 * 
	 * @param A a real matrix
	 * 
	 */
	public static void saveMatrix(String filePath, Matrix A) {
		if (A instanceof DenseMatrix) {
			saveDenseMatrix(A, filePath);
		} else {
			saveSparseMatrix(A, filePath);
		}
	}

	/**
	 * Write a dense matrix into a text file. Each line corresponds 
	 * to a row with the format "%.8g\t%.8g\t%.8g\t... \t%.8g".
	 * 
	 * @param A a dense matrix
	 * 
	 * @param filePath file path to write a dense matrix into
	 * 
	 */
	public static void saveDenseMatrix(Matrix A, String filePath) {
		PrintWriter pw = null;

		try {
			pw = new PrintWriter(new FileWriter(filePath));
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
		}

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		double[][] data = ((DenseMatrix) A).getData();
		double[] rowData = null;
		StringBuilder strBuilder = new StringBuilder(200);

		for (int i = 0; i < nRow; i++) {
			strBuilder.setLength(0);
			rowData = data[i];
			for (int j = 0; j < nCol; j++) {
				double v = rowData[j];
				int rv = (int) Math.round(v);
				if (v != rv)
					strBuilder.append(String.format("%.8g\t", v));
				else
					strBuilder.append(String.format("%d\t", rv));
			}
			// pw.printf("%g\t", A.getEntry(i, j));
			pw.println(strBuilder.toString().trim());
		}

		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data matrix file written: " + filePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}
	}

	/**
	 * Write a dense matrix into a text file. Each line corresponds 
	 * to a row with the format "%.8g\t%.8g\t%.8g\t... \t%.8g".
	 * 
	 * @param filePath file path to write a dense matrix into
	 * 
	 * @param A a dense matrix
	 * 
	 */
	public static void saveDenseMatrix(String filePath, Matrix A) {
		saveDenseMatrix(A, filePath);
	}

	/**
	 * Write a sparse matrix into a text file. Each line 
	 * corresponds to a non-zero entry (rowIdx, colIdx, value) 
	 * with the format "%d %d %.8g".
	 * 
	 * @param A a sparse matrix
	 * 
	 * @param filePath file path to write a sparse matrix into
	 * 
	 */
	public static void saveSparseMatrix(Matrix A, String filePath) {

		PrintWriter pw = null;

		try {
			pw = new PrintWriter(new FileWriter(filePath));
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
		}

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		pw.printf("numRows: %d%n", nRow);
		pw.printf("numColumns: %d%n", nCol);

		if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			int rIdx = -1;
			int cIdx = -1;
			double value = 0;
			for (int j = 0; j < nCol; j++) {
				cIdx = j + 1;
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					rIdx = ir[k] + 1;
					value = pr[k];
					if (value != 0) {
						int rv = (int) Math.round(value);
						if (value != rv)
							pw.printf("%d %d %.8g%n", rIdx, cIdx, value);
						else
							pw.printf("%d %d %d%n", rIdx, cIdx, rv);
					}
				}
			}
		}

		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data matrix file written: " + filePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}

	}

	/**
	 * Write a sparse matrix into a text file. Each line 
	 * corresponds to a non-zero entry (rowIdx, colIdx, value) 
	 * with the format "%d %d %.8g".
	 * 
	 * @param A a sparse matrix
	 * 
	 * @param filePath file path to write a sparse matrix into
	 * 
	 */
	public static void saveSparseMatrix(String filePath, Matrix A) {
		saveSparseMatrix(A, filePath);
	}

	/**
	 * Read a matrix from a text file. Sparseness will be automatically detected.
	 * 
	 * @param filePath file path to read a matrix from
	 * 
	 * @return a real matrix
	 * 
	 */
	public static Matrix loadMatrix(String filePath) {

		Matrix M = null;

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			System.out.println("Cannot open file: " + filePath);
			e.printStackTrace();
		}

		String line = "";
		int ind = 0;
		boolean isSparseMatrix = false;
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				if (Pattern.matches("numRows:[\\s]*([\\d]+)", line)) {
					isSparseMatrix = true;
					break;
				} else if (Pattern.matches("numColumns:[\\s]*([\\d]+)", line)) {
					isSparseMatrix = true;
					break;
				} else if (Pattern.matches("[(]?([\\d]+)[,] ([\\d]+)[)]?[:]? ([-\\d.]+)", line)) {
					isSparseMatrix = true;
					break;
				}
				ind++;
				if (ind == 2)
					break;
			}
			br.close();
			if (isSparseMatrix) {
				M = loadSparseMatrix(filePath);
			} else {
				M = loadDenseMatrix(filePath);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return M;

	}

	/**
	 * Read a dense matrix from a text file. Each line 
	 * corresponds to a row with the format "%.8g\t%.8g\t%.8g\t... \t%.8g".
	 * 
	 * @param filePath file path to read a dense matrix from
	 * 
	 * @return a dense matrix
	 * 
	 */
	public static DenseMatrix loadDenseMatrix(String filePath) {

		BufferedReader textIn = null;

		try {
			textIn = new BufferedReader(// Read text from a character-input stream
					new InputStreamReader(// Read bytes and decodes them into characters 
							new FileInputStream(filePath)));// Read bytes from a file
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		String line = null;

		ArrayList<double[]> denseArr = new ArrayList<double[]>();
		try {
			while ((line = textIn.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				String[] strArr = line.split("[\t ]");
				double[] vec = new double[strArr.length];
				for (int i = 0; i < strArr.length; i++) {
					vec[i] = Double.parseDouble(strArr[i]);
				}
				denseArr.add(vec);
			}
			textIn.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int nRow = denseArr.size();
		double[][] data = new double[nRow][];
		Iterator<double[]> iter = denseArr.iterator();
		int rIdx = 0;
		while (iter.hasNext()) {
			data[rIdx++] = iter.next();
		}

		return new DenseMatrix(data);

	}

	/**
	 * Load a {@code Matrix} from a text file.
	 * 
	 * @param filePath
	 *        a {@code String} specifying the location of the text file holding matrix data.
	 *        Each line is an entry with the format (without double quotes) 
	 *        "(rowIdx,[whitespace]colIdx):[whitespace]value". rowIdx and colIdx
	 *        start from 1 as in MATLAB
	 *        
	 * @return a sparse matrix
	 * 
	 */
	public static SparseMatrix loadSparseMatrix(String filePath) {

		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;

		int rIdx = 0;
		int cIdx = 0;		
		int nzmax = 0;
		double value = 0;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			System.err.println("Cannot open file: " + filePath);
			e.printStackTrace();
			System.exit(1);
		}

		int numRows = -1;
		int numColumns = -1;
		int estimatedNumRows = -1;
		int estimatedNumCols = -1;
		int ind = 0;
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				if (Pattern.matches("numRows:[\\s]*([\\d]+)", line)) {
					matcher = Pattern.compile("numRows:[\\s]*([\\d]+)").matcher(line);
					if (matcher.find()) {
						numRows = Integer.parseInt(matcher.group(1));
					}
				} else if (Pattern.matches("numColumns:[\\s]*([\\d]+)", line)) {
					matcher = Pattern.compile("numColumns:[\\s]*([\\d]+)").matcher(line);
					if (matcher.find()) {
						numColumns = Integer.parseInt(matcher.group(1));
					}
				} /*else if (Pattern.matches("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)", line)) {
					matcher = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)").matcher(line);
					if (matcher.find()) {
						rIdx = Integer.parseInt(matcher.group(1)) - 1;
						cIdx = Integer.parseInt(matcher.group(2)) - 1;
						value = Double.parseDouble(matcher.group(3));
						if (value != 0) {
							map.put(Pair.of(cIdx, rIdx), value);
							nzmax++;
						}
						if (estimatedNumRows < rIdx + 1) {
							estimatedNumRows = rIdx + 1;
						}
						if (estimatedNumCols < cIdx + 1) {
							estimatedNumCols = cIdx + 1;
						}
					}
				}*/
				ind++;
				if (ind == 2)
					break;
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				matcher = pattern.matcher(line);
				if (matcher.find()) {
					rIdx = Integer.parseInt(matcher.group(1)) - 1;
					cIdx = Integer.parseInt(matcher.group(2)) - 1;
					value = Double.parseDouble(matcher.group(3));
					if (value != 0) {
						map.put(Pair.of(cIdx, rIdx), value);
						nzmax++;
					}
					if (estimatedNumRows < rIdx + 1) {
						estimatedNumRows = rIdx + 1;
					}
					if (estimatedNumCols < cIdx + 1) {
						estimatedNumCols = cIdx + 1;
					}
				}
			}
			br.close();
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		numRows = numRows == -1 ? estimatedNumRows : numRows;
		numColumns = numColumns == -1 ? estimatedNumCols : numColumns;

		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);

	}

	/**
	 * Load a {@code Matrix} from a doc-term-count file located at
	 * {@code String} docTermCountFilePath.
	 * 
	 * @param docTermCountFilePath
	 *        a {@code String} specifying the location of the doc-term-count 
	 *        file holding matrix data. Each line is an entry with the format
	 *        (docID,[whitespace]featureID):[whitespace]value". docID
	 *        and featureID start from 1
	 *        
	 * @return a sparse matrix
	 */
	public static SparseMatrix loadMatrixFromDocTermCountFile(String docTermCountFilePath) {

		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;
		int docID = 0;
		int featureID = 0;
		double value = 0;
		int nDoc = 0;
		int nFeature = 0;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

		pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");

		try {

			br = new BufferedReader(new FileReader(docTermCountFilePath));

		} catch (FileNotFoundException e) {

			System.out.println("Cannot open file: " + docTermCountFilePath);
			e.printStackTrace();
			return null;
		} 

		int nzmax = 0;
		SparseMatrix res = null;

		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;

				matcher = pattern.matcher(line);

				if (!matcher.find()) {
					System.out.println("Data format for the docTermCountFile should be: (DocID, featureID): value");
					System.exit(0);
				}

				docID = Integer.parseInt(matcher.group(1));
				featureID = Integer.parseInt(matcher.group(2));
				value = Double.parseDouble((matcher.group(3)));

				if (nFeature < featureID)
					nFeature = featureID;

				if (nDoc < docID)
					nDoc = docID;

				if (value != 0) {
					map.put(Pair.of(docID - 1, featureID - 1), value);
					nzmax++;
				}

			}

			br.close();

			int numRows = nFeature;
			int numColumns = nDoc;
			int[] ir = new int[nzmax];
			int[] jc = new int[numColumns + 1];
			double[] pr = new double[nzmax];

			int rIdx = -1;
			int cIdx = -1;
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;

			res = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);

		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return res;

	}
	
	/**
	 * Convert a docTermCountArray instance of type {code 
	 * ArrayList<TreeMap<Integer, Integer>>} from the 
	 * TextProcessor package to a SparseMatrix instance.
	 * 
	 * @param docTermCountArray 
	 *        input where each element in the 
	 *        {@code ArrayList} is termID-count {@code TreeMap}
	 *        representation for a document
	 * 
	 * @return a sparse matrix
	 */
	public static Matrix docTermCountArray2Matrix(
			ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {

		int featureID = 0;
		double value = 0;
		int nDoc = 0;
		int nFeature = 0;
		int docID = 0;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();

		int nzmax = 0;
		Matrix res = null;

		Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
		TreeMap<Integer, Integer> feature = null;

		iter = docTermCountArray.iterator();
		while (iter.hasNext()) {
			feature = iter.next();
			docID++;
			for (int termID : feature.keySet()) {
				featureID = termID;
				value = (double)feature.get(termID);

				if (nFeature < featureID)
					nFeature = featureID;

				if (nDoc < docID)
					nDoc = docID;

				if (value != 0) {
					map.put(Pair.of(docID - 1, featureID - 1), value);
					nzmax++;
				}
			}
		}

		int numRows = nFeature;
		int numColumns = nDoc;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		res = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);

		return res;

	}
	
	/**
	 * Write a vector into a text file.
	 * 
	 * @param filePath file path to write a vector into
	 * 
	 * @param V a real vector
	 * 
	 */
	public static void save(String filePath, Vector V) {
		saveVector(filePath, V);
	}
	
	/**
	 * Write a vector into a text file.
	 * 
	 * @param V a real vector
	 * 
	 * @param filePath file path to write a vector into
	 * 
	 */
	public static void save(Vector V, String filePath) {
		saveVector(V, filePath);
	}

	/**
	 * Write a vector into a text file.
	 * 
	 * @param V a real vector
	 * 
	 * @param filePath file path to write a vector into
	 * 
	 */
	public static void saveVector(Vector V, String filePath) {
		if (V instanceof DenseVector)
			saveDenseVector((DenseVector) V, filePath);
		else if (V instanceof SparseVector)
			saveSparseVector((SparseVector) V, filePath);
	}
	
	/**
	 * Write a vector into a text file.
	 * 
	 * @param filePath file path to write a vector into
	 * 
	 * @param V a real vector
	 * 
	 */
	public static void saveVector(String filePath, Vector V) {
		saveVector(V, filePath);
	}
	
	/**
	 * Write a dense vector into a text file. Each line corresponds 
	 * to an element with the format "%.8g".
	 * 
	 * @param V a dense vector
	 * 
	 * @param filePath file path to write a dense vector into
	 * 
	 */
	public static void saveDenseVector(DenseVector V, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), true);
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
			return;
		}

		int dim = V.getDim();
		double[] pr = V.getPr();
		for (int i = 0; i < dim; i++) {
			double v = pr[i];
			int rv = (int) Math.round(v);
			if (v != rv)
				pw.printf("%.8g%n", v);
			else
				pw.printf("%d%n", rv);
		}

		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data vector file written: " + filePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}
	}
	
	/**
	 * Write a sparse vector into a text file. Each line 
	 * corresponds to a non-zero element (index, value) 
	 * with the format "%d %.8g".
	 * 
	 * @param V a sparse vector
	 * 
	 * @param filePath file path to write a sparse vector into
	 * 
	 */
	public static void saveSparseVector(SparseVector V, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), true);
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
			return;
		}

		int dim = V.getDim();
		pw.printf("dim: %d%n", dim);

		int[] ir = V.getIr();
		double[] pr = V.getPr();
		int nnz = V.getNNZ();
		int idx = -1;
		double val = 0;
		for (int k = 0; k < nnz; k++) {
			idx = ir[k] + 1;
			val = pr[k];
			if (val != 0) {
				int rv = (int) Math.round(val);
				if (val != rv)
					pw.printf("%d %.8g%n", idx, val);
				else
					pw.printf("%d %d%n", idx, rv);
			}
		}

		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data vector file written: " + filePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}
	}
	
	/**
	 * Write a dense vector into a text file. Each line corresponds 
	 * to an element with the format "%.8g".
	 * 
	 * @param filePath file path to write a dense vector into
	 * 
	 * @param V a dense vector
	 * 
	 */
	public static void saveDenseVector(String filePath, DenseVector V) {
		saveDenseVector(V, filePath);
	}
	
	/**
	 * Write a sparse vector into a text file. Each line 
	 * corresponds to a non-zero element (index, value) 
	 * with the format "%d %.8g".
	 * 
	 * @param filePath file path to write a sparse vector into
	 * 
	 * @param V a sparse vector
	 * 
	 */
	public static void saveSparseVector(String filePath, SparseVector V) {
		saveSparseVector(V, filePath);
	}
	
	/**
	 * Read a vector from a text file. Sparseness will be automatically detected.
	 * 
	 * @param filePath file path to read a vector from
	 * 
	 * @return a real vector
	 * 
	 */
	public static Vector loadVector(String filePath) {
		
		Vector V = null;

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			System.out.println("Cannot open file: " + filePath);
			e.printStackTrace();
			return null;
		}

		String line = "";
		int ind = 0;
		boolean isSparseVector = false;
		try {
			while ((line = br.readLine()) != null) {
				if (line.startsWith("#") || line.trim().isEmpty())
					continue;
				if (Pattern.matches("dim:[\\s]*([\\d]+)", line)) {
					isSparseVector = true;
					break;
				}
				ind++;
				if (ind == 1)
					break;
			}
			br.close();
			if (isSparseVector) {
				V = loadSparseVector(filePath);
			} else {
				V = loadDenseVector(filePath);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return V;
		
	}
	
	/**
	 * Read a dense vector from a text file. Each line 
	 * corresponds to an element with the format "%.8g".
	 * 
	 * @param filePath file path to read a dense vector from
	 * 
	 * @return a dense vector
	 * 
	 */
	public static DenseVector loadDenseVector(String filePath) {

		BufferedReader textIn = null;

		try {
			textIn = new BufferedReader(// Read text from a character-input stream
					new InputStreamReader(// Read bytes and decodes them into characters 
							new FileInputStream(filePath)));// Read bytes from a file
		} catch (FileNotFoundException e) {
			System.out.println("Cannot open file: " + filePath);
			e.printStackTrace();
			return null;
		}

		String line = null;

		ArrayList<Double> denseArr = new ArrayList<Double>();
		try {
			while ((line = textIn.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				denseArr.add(Double.parseDouble(line));
			}
			textIn.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int dim = denseArr.size();
		double[] pr = new double[dim];
		Iterator<Double> iter = denseArr.iterator();
		int idx = 0;
		while (iter.hasNext()) {
			pr[idx++] = iter.next();
		}

		return new DenseVector(pr);
		
	}
	
	/**
	 * Read a sparse vector from a text file. Each line 
	 * corresponds to a non-zero element (index, value) 
	 * with the format "%.8g".
	 * 
	 * @param filePath file path to read a sparse vector from
	 * 
	 * @return a sparse vector
	 * 
	 */
	public static SparseVector loadSparseVector(String filePath) {

		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;

		int idx = 0;
		int nnz = 0;
		double val = 0;
		TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();

		try {
			br = new BufferedReader(new FileReader(filePath));
		} catch (FileNotFoundException e) {
			System.err.println("Cannot open file: " + filePath);
			e.printStackTrace();
			// System.exit(1);
			return null;
		}

		int dim = -1;
		int estimatedDim = -1;
		int ind = 0;
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				if (Pattern.matches("dim:[\\s]*([\\d]+)", line)) {
					matcher = Pattern.compile("dim:[\\s]*([\\d]+)").matcher(line);
					if (matcher.find()) {
						dim = Integer.parseInt(matcher.group(1));
					}
				}
				ind++;
				if (ind == 1)
					break;
			}
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		pattern = Pattern.compile("[(]?([\\d]+)[)]?[:]? ([-\\d.]+)");
		try {
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("#") || line.isEmpty())
					continue;
				matcher = pattern.matcher(line);
				if (matcher.find()) {
					idx = Integer.parseInt(matcher.group(1)) - 1;
					val = Double.parseDouble(matcher.group(2));
					if (val != 0) {
						map.put(idx, val);
						nnz++;
					}
					if (estimatedDim < idx + 1) {
						estimatedDim = idx + 1;
					}
				}
			}
			br.close();
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		dim = dim == -1 ? estimatedDim : dim;
		
		int[] ir = new int[nnz];
		double[] pr = new double[nnz];

		int k = 0;
		for (Entry<Integer, Double> entry : map.entrySet()) {
			idx = entry.getKey();
			ir[k] = idx;
			pr[k] = entry.getValue();
			k++;
		}

		return new SparseVector(ir, pr, nnz, dim);
		
	}
	
}
