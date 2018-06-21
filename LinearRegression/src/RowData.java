import java.util.Scanner;

/**
 * Object to obtain and store values for row data.
 * 
 * @author cyrilyared
 *
 */
public class RowData {
	
	double[] data;
	private Scanner userInput;
	
	public RowData(String prompt, int sizeTrainingSet) {
		data = new double[sizeTrainingSet];
		getUserValues(prompt, sizeTrainingSet);
	}

	/**
	 * Obtains comma-separated values from user for row data.
	 * Removes spaces.
	 * Parses values as double and inserts values into data array.
	 * 
	 * @param prompt String message to display
	 * @param sizeTrainingSet integer size of training set
	 */
	private void getUserValues(String prompt, int sizeTrainingSet) {
		boolean error = true;
		userInput = new Scanner(System.in);
		
		while(error) {
			error = false;
			System.out.println(prompt);
			String input = userInput.nextLine();
			input = input.replaceAll("\\s+", "");
			String inputSeparated[] = input.split(",");
			
			if(inputSeparated.length != sizeTrainingSet) {
				error = true;
				continue;
			}
			
			for(int i = 0; i < sizeTrainingSet; i++) {
				try {
					data[i] = Double.parseDouble(inputSeparated[i]);
				} catch(NumberFormatException e) {
					error = true;
				}
			}
		}
	}
}