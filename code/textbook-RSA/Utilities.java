import java.math.BigInteger;

public class Utilities {

	public static BigInteger[] toAscii( String message ) {
		BigInteger[] ascii = new BigInteger[message.length()];
		int i = 0;
		// convert each character of the original message
		// into its corresponding ascii code
		for ( char c : message.toCharArray() ) {
			int code = Integer.valueOf( (int) c );
			ascii[i] = new BigInteger( Integer.toString( code ) );
			i++;
		}
		i = 0;
		return ascii;
	}

}
