import java.math.BigInteger;
//Today I added this line :)
//Hello from Git !!!
public class Main {

	public static void main( String[] args ) {

		String message = "Encrypt it yourself!";
		Keys keys = new Keys();

		String encryptedMessage = RSA.encrypt( keys.getPublicKey(), message );
		String decryptedMessage = RSA.decrypt( keys.getPrivateKey(), encryptedMessage );

		// public key components
		BigInteger e = keys.getPublicKey()[0];
		BigInteger n = keys.getPublicKey()[1];

		// private key components
		BigInteger d = keys.getPrivateKey()[0];
		BigInteger m = keys.getPrivateKey()[1]; // == n

		System.out.println( "Public key: ( e = " + e + ", n = " + n + ")" );
		System.out.println( "Private key: ( d = " + d + ", n = " + m + ")" );
		System.out.println( "/-------------------------------/" );

		System.out.println( "Original message: " + message );
		System.out.println( "Encrypted message: " + encryptedMessage );
		System.out.println( "Decrypted message: " + decryptedMessage );

	}

}
