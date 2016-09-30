package la.utils;

import static la.io.IO.saveMatrix;
import static la.io.IO.saveVector;
import static la.utils.Utility.exit;
import static la.utils.Printer.sprintf;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import la.matrix.DenseMatrix;
import la.vector.DenseVector;

public class IO {

	/**
	 * Write a 2D {@code double} array into a text file.
	 * 
	 * @param filePath file path to write a 2D {@code double} array into
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 */
	public static void save(String filePath, double[][] A) {
		saveMatrix(filePath, new DenseMatrix(A));
	}
	
	/**
	 * Write a 2D {@code double} array into a text file.
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param filePath file path to write a 2D {@code double} array into
	 * 
	 */
	public static void save(double[][] A, String filePath) {
		saveMatrix(new DenseMatrix(A), filePath);
	}
	
	/**
	 * Write a 1D {@code double} array into a text file.
	 * 
	 * @param filePath file path to write a 1D {@code double} array into
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void save(String filePath, double[] V) {
		saveVector(filePath, new DenseVector(V));
	}
	
	/**
	 * Write a 1D {@code double} array into a text file.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param filePath file path to write a 1D {@code double} array into
	 * 
	 */
	public static void save(double[] V, String filePath) {
		saveVector(new DenseVector(V), filePath);
	}
	
	/**
	 * Write a 1D {@code double} array into a text file.
	 * 
	 * @param filePath file path to write a 1D {@code double} array into
	 * 
	 * @param V a 1D integer array
	 */
	public static void save(String filePath, int[] V) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), true);
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
			return;
		}
		
		for (int i = 0; i < V.length; i++) {
			pw.printf("%d%n", V[i]);
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
	 * Write a 1D {@code double} array into a text file.
	 * 
	 * @param V a 1D integer array
	 * 
	 * @param filePath file path to write a 1D {@code double} array into
	 */
	public static void save(int[] V, String filePath) {
		save(filePath, V);
	}
	
	/**
	 * Save a {@code Map<K, V>} in a text file.
	 * 
	 * @param map a {@code Map<K, V>} instance
	 * 
	 * @param filePath file path to write into
	 */
	public static <K, V> void saveMap(Map<K, V> map, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		for (Entry<K, V> entry : map.entrySet()) {
			pw.print(entry.getKey());
			pw.print('\t');
			pw.println(entry.getValue());
		}
		pw.close();
	}
	
	/**
	 * Save a {@code List<V>} in a text file.
	 * 
	 * @param list a {@code List<V>} instance
	 * 
	 * @param filePath file path to write into
	 */
	public static <V> void saveList(List<V> list, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		for (V v : list) {
			pw.println(v);
		}
		pw.close();
	}
	
	/**
	 * Save a string in a text file.
	 * 
	 * @param filePath file path to write into
	 * 
	 * @param content string content
	 */
	public static void saveString(String filePath, String content) {
		PrintWriter pw = null;
		boolean autoFlush = true;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), autoFlush);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		pw.print(content);
		pw.close();
	}
	
	/**
	 * Save a string in a text file.
	 * 
	 * @param filePath file path to write into
	 * 
	 * @param content string content
	 */
	public static void save(String filePath, String content) {
		saveString(filePath, content);
	}
	
	/**
	 * Save specification for each vector element.
	 * 
	 * @param V a dense vector
	 * 
	 * @param spec specification strings for all elements
	 * 
	 * @param filePath file path to save the specifications
	 */
	public static void saveSpec(DenseVector V, String[] spec, String filePath) {
		saveSpec(V, spec, 4, filePath);
	}
	
	/**
	 * Save specification for each vector element.
	 * 
	 * @param V a dense vector
	 * 
	 * @param spec specification strings for all elements
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 * @param filePath file path to save the specifications
	 */
	public static void saveSpec(DenseVector V, String[] spec, int p, String filePath) {
		PrintWriter pw = null;
		boolean autoFlush = true;
		try {
			pw = new PrintWriter(
					new BufferedWriter(
							new FileWriter(filePath)), autoFlush);
		} catch (IOException e) {
			e.printStackTrace();
			exit(1);
		}
		if (V instanceof DenseVector) {
			int dim = V.getDim();
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < dim; k++) {
				pw.print("  ");
				double v = pr[k];
				int rv = (int) Math.round(v);
				String valueString;
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				pw.println(sprintf(sprintf("%%%ds  %%s", 8 + p - 4), valueString, spec[k]));
			}
			pw.println();
		} else {
			System.err.println("The input vector should be a DenseVector instance");
			exit(1);
		}
		pw.close();
	}
	
}
