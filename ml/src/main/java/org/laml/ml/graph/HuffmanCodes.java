package org.laml.ml.graph;

import static org.laml.la.utils.Printer.println;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import org.laml.la.utils.BinaryNode;
import org.laml.la.utils.Entry;
import org.laml.la.utils.PriorityQueue;


/**
 * A Java implementation of Huffman codes. Since a Huffman tree
 * is a full binary tree, and each character is only at the leaf,
 * thus no character is the parent of another character, so Huffman
 * tree is a prefix tree.
 * 
 * @author Mingjie Qian
 * @version 1.0 April 17th, 2016
 */
public class HuffmanCodes {

	public static void main(String[] args) {
		char[] characters = {'a', 'b', 'c', 'd', 'e', 'f'};
		int[] freqencies = {45, 13, 12, 16, 9, 5};
		BinaryNode<Entry<Integer, Character>> HuffmanTree = buildHuffmanTree(characters, freqencies);
		HuffmanCodeTable HuffmanCodeTable = generateHuffmanCodeTable(HuffmanTree);
		println("Encoder:");
		println(HuffmanCodeTable.encoder);
		println("Decoder:");
		println(HuffmanCodeTable.decoder);
		String text = "aabcdfeabdfebaccbed";
		println("Text:");
		println(text);
		String encodedString = HuffmanCodeTable.encode(text);
		println("Encoded string:");
		println(encodedString);
		String decodedString = HuffmanCodeTable.decode(encodedString);
		println("Decoded string:");
		println(decodedString);
	}
	
	private static HuffmanCodeTable generateHuffmanCodeTable(
			BinaryNode<Entry<Integer, Character>> HuffmanTree) {
		Map<Character, String> encoder = new HashMap<Character, String>();
		Map<String, Character> decoder = new HashMap<String, Character>();
		StringBuilder sb = new StringBuilder();
		buildHuffmanCodeTable(HuffmanTree, sb, encoder, decoder);
		return new HuffmanCodeTable(HuffmanTree, encoder, decoder);
	}
	
	private static void buildHuffmanCodeTable(
			BinaryNode<Entry<Integer, Character>> root,
			StringBuilder sb,
			Map<Character, String> encoder,
			Map<String, Character> decoder) {
		/*
		 * Huffman Tree must be a full tree meaning that a node is
		 * either an inner node or a leaf.
		 */
		if (root.left == null && root.right == null) {
			String codeword = sb.toString();
			Character c = root.key.value;
			encoder.put(c, codeword);
			decoder.put(codeword, c);
			return;
		}
		sb.append('0');
		buildHuffmanCodeTable(root.left, sb, encoder, decoder);
		sb.setLength(sb.length() - 1);
		
		sb.append('1');
		buildHuffmanCodeTable(root.right, sb, encoder, decoder);
		sb.setLength(sb.length() - 1);
	}

	static class HuffmanCodeTable {
		BinaryNode<Entry<Integer, Character>> HuffmanTree;
		public Map<Character, String> encoder;
		public Map<String, Character> decoder;
		public HuffmanCodeTable(BinaryNode<Entry<Integer, Character>> HuffmanTree, Map<Character, String> encoder, Map<String, Character> decoder) {
			this.HuffmanTree = HuffmanTree;
			this.encoder = encoder;
			this.decoder = decoder;
		}
		public String encode(String text) {
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < text.length(); i++)
				sb.append(encoder.get(text.charAt(i)));
			return sb.toString();
		}
		public String decode(String encodeString) {
			StringBuilder sb = new StringBuilder();
			int i = 0;
			while (i < encodeString.length()) {
				BinaryNode<Entry<Integer, Character>> node = HuffmanTree;
				while (node.left != null) {
					if (encodeString.charAt(i++) == '0') {
						node = node.left;
					} else
						node = node.right;
				}
				sb.append(node.key.value);
			}
			return sb.toString();
		}
	}

	private static BinaryNode<Entry<Integer, Character>> buildHuffmanTree(char[] characters,
			int[] freqencies) {
		PriorityQueue<BinaryNode<Entry<Integer, Character>>> queue = new PriorityQueue<BinaryNode<Entry<Integer, Character>>>(
				characters.length,
				new Comparator<BinaryNode<Entry<Integer, Character>>>() {
					@Override
					public int compare(BinaryNode<Entry<Integer, Character>> o1, BinaryNode<Entry<Integer, Character>> o2) {
						int diff = o2.key.key - o1.key.key;
						return diff == 0 ? 0 : diff < 0 ? -1 : 1;
					}
				}
				);
		for (int i = 0; i < characters.length; i++) {
			queue.insert(new BinaryNode<Entry<Integer, Character>>(new Entry<Integer, Character>(freqencies[i], characters[i])));
		}
		BinaryNode<Entry<Integer, Character>> HuffmanTree = null;
		for (int i = 1; i < characters.length; i++) {
			BinaryNode<Entry<Integer, Character>> x = queue.poll();
			BinaryNode<Entry<Integer, Character>> y = queue.poll();
			BinaryNode<Entry<Integer, Character>> z = new BinaryNode<Entry<Integer, Character>>(new Entry<Integer, Character>(x.key.key + y.key.key, null));
			z.left = x;
			z.right = y;
			queue.insert(z);
		}
		HuffmanTree = queue.poll();
		return HuffmanTree;
	}

}
